// SPDX-License-Identifier: GPL-3.0-or-later
#include "nvn_shader.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/environment.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/ir/modifiers.h"
#include "shader_recompiler/frontend/maxwell/control_flow.h"
#include "shader_recompiler/frontend/maxwell/translate_program.h"
#include "shader_recompiler/host_translate_info.h"
#include "shader_recompiler/object_pool.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

namespace nvnshader {

namespace {

class NvnEnvironment final : public Shader::Environment {
public:
    NvnEnvironment(const uint8_t* code, size_t max_size, const Options& options, bool compute)
        // En compute NVN deja toda la tabla del driver en c0 (handles en 0x168 + 8·i, SSBO en 0x310 + 16·k):
        // los TEX con índice inmediato leen ahí, no en c2 como en gráficos.
        : code_(code), max_size_(max_size), texture_bound_(compute ? 0 : options.texture_bound_buffer), options_(options) {
        if (compute) {
            stage = Shader::Stage::Compute;
            start_address = options.compute_code_offset;
            return;
        }
        start_address = kNvnHeaderSize;
        std::memcpy(&sph, code_ + kNvnHeaderSize, sizeof(sph));
        switch (sph.common0.shader_type) {
            case 1: stage = Shader::Stage::VertexB; break;
            case 2: stage = Shader::Stage::TessellationControl; break;
            case 3: stage = Shader::Stage::TessellationEval; break;
            case 4: stage = Shader::Stage::Geometry; break;
            case 5: stage = Shader::Stage::Fragment; break;
            default: stage = Shader::Stage::Compute; break;
        }
    }

    u64 ReadInstruction(u32 address) override {
        u64 v = 0;
        if (address + sizeof v <= max_size_) std::memcpy(&v, code_ + address, sizeof v);
        return v;
    }
    // Los valores de los constant buffers solo existen al dibujar, salvo c1 (constantes del propio programa):
    // ahí están las tablas de salto de los BRX, que Eden necesita para encontrar los destinos del switch.
    u32 ReadCbufValue(u32 index, u32 offset) override {
        if (index == 1) brx_ = true;  // Eden solo lee c1 al traducir para las tablas de un BRX (switch)
        u32 v = 0;
        if (index == 1 && options_.constants && offset + sizeof v <= options_.constants_size)
            std::memcpy(&v, options_.constants + offset, sizeof v);
        return v;
    }
    bool HasBrx() const { return brx_; }
    Shader::TextureType ReadTextureType(u32) override { return Shader::TextureType::Color2D; }
    Shader::TexturePixelFormat ReadTexturePixelFormat(u32) override {
        return Shader::TexturePixelFormat::A8B8G8R8_UNORM;
    }
    bool IsTexturePixelFormatInteger(u32) override { return false; }
    u32 ReadViewportTransformState() override { return 1; }
    u32 TextureBoundBuffer() const override { return texture_bound_; }
    // Gráficos: lo dice la cabecera (SPH), como en Eden con un GPU de verdad (más la pila CRS). Compute no
    // tiene SPH: lo que diga la sección de control o, si no, un margen (solo cuenta si el shader la usa).
    u32 LocalMemorySize() const override {
        if (stage != Shader::Stage::Compute)
            return static_cast<u32>(sph.LocalMemorySize()) + sph.common3.shader_local_memory_crs_size;
        return options_.local_memory ? options_.local_memory : 1024;
    }
    u32 SharedMemorySize() const override { return options_.shared_memory; }
    std::array<u32, 3> WorkgroupSize() const override {
        return {options_.workgroup[0], options_.workgroup[1], options_.workgroup[2]};
    }
    // Parámetros del draw que el driver de NVN deja en su c0 (las macros que Eden emula): base vertex en 0x640, base
    // instance en 0x644 y el número de draw de un multi-draw en 0x648 (gl_BaseVertexARB, gl_BaseInstanceARB y
    // gl_DrawIDARB del GLSL de NVN), que salen de Vulkan (gl_BaseVertex, gl_BaseInstance, gl_DrawID). MH Rise usa 0x640
    // como VertexId − c0[0x640] (traza tf1, draws 14, 182 y 192): el índice sin base. VertexId es gl_VertexIndex, que
    // lleva el vertexOffset que usemos (desplazado a la ventana de vértices en los draws directos, el del juego en los
    // indirectos): restar gl_BaseVertex lo quita en todos los casos. Con 0x640 a cero (antes) un draw directo desplazado
    // leía el vértice índice − mínimo y un multi-draw con base vertex 1302, el 1302 + i.
    bool HasHLEMacroState() const override { return stage == Shader::Stage::VertexB; }
    std::optional<Shader::ReplaceConstant> GetReplaceConstBuffer(u32 bank, u32 offset) override {
        if (bank != 0 || stage != Shader::Stage::VertexB) return std::nullopt;
        if (offset == 0x640) return Shader::ReplaceConstant::BaseVertex;
        if (offset == 0x644) return Shader::ReplaceConstant::BaseInstance;
        if (offset == 0x648) return Shader::ReplaceConstant::DrawID;
        return std::nullopt;
    }
    void Dump(u64, u64) override {}

private:
    const uint8_t* code_;
    bool brx_ = false;
    size_t max_size_;
    uint32_t texture_bound_;
    const Options& options_;
};

Stage to_stage(Shader::Stage s) {
    switch (s) {
        case Shader::Stage::VertexB: return Stage::Vertex;
        case Shader::Stage::TessellationControl: return Stage::TessControl;
        case Shader::Stage::TessellationEval: return Stage::TessEval;
        case Shader::Stage::Geometry: return Stage::Geometry;
        case Shader::Stage::Fragment: return Stage::Fragment;
        default: return Stage::Compute;
    }
}

// Mismo orden y mismo contador que EmitContext con bindings unificados y sin aliasing de
// descriptores: UBOs y SSBOs avanzan `count`, el resto uno por descriptor.
void list_resources(const Shader::Info& info, uint32_t binding, Translated& out) {
    for (const auto& d : info.constant_buffer_descriptors) {
        for (uint32_t k = 0; k < d.count; k++)
            out.resources.push_back({Resource::UniformBuffer, binding++, d.index + k, 0, 0});
    }
    for (const auto& d : info.storage_buffers_descriptors) {
        for (uint32_t k = 0; k < d.count; k++) {
            Resource r{Resource::StorageBuffer, binding++, d.cbuf_index, d.cbuf_offset + k * 16, 0};
            r.is_written = d.is_written;
            out.resources.push_back(r);
        }
    }
    for (const auto& d : info.texture_buffer_descriptors)
        out.resources.push_back({Resource::TextureBuffer, binding++, d.cbuf_index, d.cbuf_offset, 0});
    for (const auto& d : info.image_buffer_descriptors)
        out.resources.push_back({Resource::ImageBuffer, binding++, d.cbuf_index, d.cbuf_offset, 0});
    for (const auto& d : info.texture_descriptors) {
        Resource r{Resource::Texture, binding++, d.cbuf_index, d.cbuf_offset, static_cast<uint32_t>(d.type)};
        r.is_depth = d.is_depth;
        r.has_secondary = d.has_secondary;
        r.secondary_cbuf_index = d.secondary_cbuf_index;
        r.secondary_cbuf_offset = d.secondary_cbuf_offset;
        r.shift_left = d.shift_left;
        r.secondary_shift_left = d.secondary_shift_left;
        r.count = d.count;
        r.size_shift = d.size_shift;
        r.handle_and_mask = d.handle_and_mask;
        r.handle_or = d.handle_or;
        out.resources.push_back(r);
    }
    for (const auto& d : info.image_descriptors) {
        Resource r{Resource::Image, binding++, d.cbuf_index, d.cbuf_offset, static_cast<uint32_t>(d.type)};
        r.is_written = d.is_written;
        out.resources.push_back(r);
    }
}

// Los arrays de texturas (count > 1) al set 1, con bindings etapa · 16 + j en el orden de los recursos: en el SPIR-V se
// cambian sus decoraciones Binding y DescriptorSet. Así el layout del set 1 depende solo de cuántos arrays hay y de qué
// tamaño, y dos shaders que leen los mismos handles comparten el set (el ejecutor lo arma una vez por envío: 3 arrays de
// 512 y uno de 223 por pixel shader de MH Rise no se pueden escribir en cada draw).
void move_arrays_to_bindless_set(Translated& out) {
    std::vector<std::pair<uint32_t, uint32_t>> moved;  // (binding de Eden, binding en el set 1)
    const uint32_t first = static_cast<uint32_t>(out.stage) * kBindlessPerStage;
    for (Resource& r : out.resources)
        if (r.kind == Resource::Texture && r.count > 1 && moved.size() < kBindlessPerStage) {
            moved.push_back({r.binding, first + static_cast<uint32_t>(moved.size())});
            r.binding = moved.back().second;
            r.set = kBindlessSet;
        }
    if (moved.empty()) return;
    out.has_bindless_set = true;
    constexpr uint32_t kOpDecorate = 71, kBinding = 33, kDescriptorSet = 34;
    std::vector<uint32_t> targets;  // variables movidas
    std::vector<uint32_t>& w = out.spirv;
    for (int pass = 0; pass < 2; pass++)
        for (size_t k = 5; k < w.size();) {
            const uint32_t count = w[k] >> 16, op = w[k] & 0xFFFF;
            if (!count) break;
            if (op == kOpDecorate && count >= 4) {
                const uint32_t target = w[k + 1], decoration = w[k + 2];
                if (pass == 0 && decoration == kBinding)
                    for (const auto& [from, to] : moved)
                        if (w[k + 3] == from) {
                            w[k + 3] = to;
                            targets.push_back(target);
                            break;
                        }
                if (pass == 1 && decoration == kDescriptorSet &&
                    std::find(targets.begin(), targets.end(), target) != targets.end())
                    w[k + 3] = kBindlessSet;
            }
            k += count;
        }
}

// Valor de una expresión entera del IR que solo depende de LocalInvocationId y de inmediatos, para la invocación lid.
// Falso si depende de cualquier otra cosa (cbufs, memoria, phis…) o se sale de lo que se sabe evaluar.
bool eval_invocation(const Shader::IR::Value& value, const uint32_t lid[3], uint32_t& out, int depth = 0) {
    using Shader::IR::Opcode;
    if (depth > 64) return false;
    const Shader::IR::Value v = value.Resolve();
    if (v.IsImmediate()) {
        if (v.Type() != Shader::IR::Type::U32) return false;
        out = v.U32();
        return true;
    }
    const Shader::IR::Inst* inst = v.Inst();
    const Opcode op = inst->GetOpcode();
    if (op == Opcode::CompositeExtractU32x3) {
        const Shader::IR::Value vec = inst->Arg(0).Resolve(), index = inst->Arg(1).Resolve();
        if (vec.IsImmediate() || vec.Inst()->GetOpcode() != Opcode::LocalInvocationId || !index.IsImmediate() ||
            index.U32() > 2)
            return false;
        out = lid[index.U32()];
        return true;
    }
    uint32_t a = 0, b = 0, c = 0;
    const size_t n = inst->NumArgs();
    if (n < 2 || n > 3 || !eval_invocation(inst->Arg(0), lid, a, depth + 1) ||
        !eval_invocation(inst->Arg(1), lid, b, depth + 1) || (n == 3 && !eval_invocation(inst->Arg(2), lid, c, depth + 1)))
        return false;
    switch (op) {
        case Opcode::IAdd32: out = a + b; return true;
        case Opcode::ISub32: out = a - b; return true;
        case Opcode::IMul32: out = a * b; return true;
        case Opcode::BitwiseAnd32: out = a & b; return true;
        case Opcode::BitwiseOr32: out = a | b; return true;
        case Opcode::BitwiseXor32: out = a ^ b; return true;
        case Opcode::ShiftLeftLogical32: out = a << b; return b < 32;
        case Opcode::ShiftRightLogical32: out = a >> b; return b < 32;
        case Opcode::ShiftRightArithmetic32: out = static_cast<uint32_t>(static_cast<int32_t>(a) >> b); return b < 32;
        case Opcode::BitFieldUExtract:
            if (b + c > 32) return false;
            out = c == 0 ? 0 : (a >> b) & (c == 32 ? ~0u : (1u << c) - 1);
            return true;
        default: return false;
    }
}

// Memoria compartida que en realidad es privada de cada invocación: el compilador de NVN derrama registros a la memoria
// compartida (STS/LDS en tid·28 + k en el skinning de la cara de MH Rise, 793657b4f1e65631: el índice del 4.º hueso y
// las componentes que no le caben en registros). Si todos los accesos son base(invocación) + k, con la misma base para
// la invocación en todos y ventanas [base + kmin, base + kmax + 4) que no se solapan entre las invocaciones del grupo,
// ninguna lee lo que escribe otra (con o sin barreras): es memoria privada y se pasa a la memoria local de Eden
// (Private con índices constantes, que el driver deja en registros). Con esos stores a Workgroup dentro de los casos de
// su switch (BRX), el driver de Qualcomm (512.746) deja la salida de la Flip igual, bit a bit, a la de ese SPIR-V con los
// 4 pesos del skinning a 0 (los del caso 1272, que los pone a 0): suma 0, 1/0 = inf, inf·0 = NaN en todo el grupo; un
// intérprete de SPIR-V con los mismos buffers da posiciones finitas. El skinning hermano sin memoria compartida
// (6aaf54a71876a3f2, mismo switch y mismos casos) sale bien. Devuelve los accesos cambiados (0: no aplica).
int shared_to_local(Shader::IR::Program& program, const std::array<u32, 3>& workgroup) {
    using Shader::IR::Opcode;
    if (program.stage != Shader::Stage::Compute || program.shared_memory_size == 0) return 0;
    struct Access {
        Shader::IR::Inst* inst;
        Shader::IR::Value base;
        uint32_t imm;
    };
    std::vector<Access> accesses;
    bool uses_local = false;
    for (Shader::IR::Block* block : program.post_order_blocks)
        for (Shader::IR::Inst& inst : block->Instructions()) {
            const Opcode op = inst.GetOpcode();
            if (op == Opcode::LoadLocal || op == Opcode::WriteLocal) uses_local = true;
            if (op != Opcode::LoadSharedU32 && op != Opcode::WriteSharedU32) {
                if (Shader::IR::NameOf(op).find("Shared") != std::string_view::npos) return 0;  // otros tamaños, atómicos
                continue;
            }
            const Shader::IR::Value offset = inst.Arg(0).Resolve();
            Shader::IR::Value base = offset;
            uint32_t imm = 0;
            if (!offset.IsImmediate() && offset.Inst()->GetOpcode() == Opcode::IAdd32) {
                const Shader::IR::Value x = offset.Inst()->Arg(0).Resolve(), y = offset.Inst()->Arg(1).Resolve();
                if (y.IsImmediate() && y.Type() == Shader::IR::Type::U32) base = x, imm = y.U32();
                else if (x.IsImmediate() && x.Type() == Shader::IR::Type::U32) base = y, imm = x.U32();
            }
            if (imm % 4 || imm >= 4096) return 0;
            accesses.push_back({&inst, base, imm});
        }
    if (accesses.empty()) return 0;
    uint32_t lo = ~0u, hi = 0;
    for (const Access& a : accesses) lo = std::min(lo, a.imm), hi = std::max(hi, a.imm + 4);
    const uint32_t window = hi - lo;
    std::vector<uint32_t> bases;
    for (uint32_t z = 0; z < workgroup[2]; z++)
        for (uint32_t y = 0; y < workgroup[1]; y++)
            for (uint32_t x = 0; x < workgroup[0]; x++) {
                const uint32_t lid[3] = {x, y, z};
                uint32_t first = 0;
                for (size_t k = 0; k < accesses.size(); k++) {
                    uint32_t b = 0;
                    if (!eval_invocation(accesses[k].base, lid, b) || (k && b != first)) return 0;
                    first = b;
                }
                // Fuera de la memoria compartida declarada (lo que haría el original es indefinido): no se toca.
                if (uint64_t{first} + hi > program.shared_memory_size) return 0;
                bases.push_back(first);
            }
    std::sort(bases.begin(), bases.end());
    for (size_t k = 1; k < bases.size(); k++)
        if (bases[k] - bases[k - 1] < window) return 0;
    // La ventana va tras la memoria local que ya use el programa (LDL/STL), o sola (la de 1 KB por defecto no se usaba).
    const uint32_t first_word = uses_local ? (program.local_memory_size + 3) / 4 : 0;
    for (const Access& a : accesses) {
        const bool load = a.inst->GetOpcode() == Opcode::LoadSharedU32;
        a.inst->ReplaceOpcode(load ? Opcode::LoadLocal : Opcode::WriteLocal);
        a.inst->SetArg(0, Shader::IR::Value{first_word + (a.imm - lo) / 4});
    }
    program.local_memory_size = first_word * 4 + window;
    program.shared_memory_size = 0;
    program.info.uses_local_memory = true;
    return static_cast<int>(accesses.size());
}

}  // namespace

uint32_t binding_base(Stage s) { return s == Stage::Compute ? 0 : static_cast<uint32_t>(s) * kBindingsPerStage; }

const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Vertex: return "vertex";
        case Stage::TessControl: return "tess-control";
        case Stage::TessEval: return "tess-eval";
        case Stage::Geometry: return "geometry";
        case Stage::Fragment: return "pixel";
        case Stage::Compute: return "compute";
    }
    return "?";
}

bool translate(const uint8_t* code, size_t max_size, const Options& options, Translated& out) {
    out = {};
    uint32_t magic = 0;
    if (max_size >= 4) std::memcpy(&magic, code, 4);
    const bool compute = magic == kNvnComputeCodeMagic || (options.compute_hint && magic != kNvnCodeMagic);
    if ((magic != kNvnCodeMagic && !compute) || max_size < kNvnHeaderSize + sizeof(Shader::ProgramHeader)) {
        out.error = "no es código NVN";
        return false;
    }
    try {
        NvnEnvironment env(code, max_size, options, compute);
        out.stage = to_stage(env.ShaderStage());
        Shader::ObjectPool<Shader::Maxwell::Flow::Block> flow_blocks;
        Shader::ObjectPool<Shader::IR::Inst> insts;
        Shader::ObjectPool<Shader::IR::Block> blocks;
        Shader::Maxwell::Flow::CFG cfg(env, flow_blocks, env.StartAddress() + (compute ? 0 : sizeof(Shader::ProgramHeader)));
        Shader::HostTranslateInfo host{};
        host.support_conditional_barrier = options.keep_conditional_barriers;
        // Bindless de los uniform buffers del juego: arrays hasta el final del cbuf (los decals de c9 son 256 registros de
        // 128 bytes en 32 KB; con el límite de 1024 descriptores repartido entre los arrays se cortaban). El Adreno admite
        // 524288 por etapa.
        host.bindless_cbuf_arrays = options.bindless_arrays;
        if (options.bindless_arrays)
            host.max_per_stage_descriptor_sampled_images = host.max_descriptor_set_sampled_images = host.max_per_stage_resources = 4096;
        ShimLog::conditional_barriers = 0;
        Shader::IR::Program program = Shader::Maxwell::TranslateProgram(insts, blocks, env, cfg, host);
        out.conditional_barriers = ShimLog::conditional_barriers;
        if (options.shared_to_local) out.shared_to_local = shared_to_local(program, env.WorkgroupSize());
        if (options.ir) {
            *options.ir = Shader::IR::DumpProgram(program);
            // Qué textura (índice de descriptor) usa cada instrucción de imagen: el volcado de Eden no lo dice.
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (Shader::IR::NameOf(inst.GetOpcode()).starts_with("Image")) {
                        char line[96];
                        snprintf(line, sizeof line, "[%016llx] descriptor %u\n",
                                 static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(&inst)),
                                 static_cast<unsigned>(inst.Flags<Shader::IR::TextureInstInfo>().descriptor_index));
                        *options.ir += line;
                    }
        }

        // DEPURACIÓN: NIER_SIN_DESCARTE: los pixel shaders sin sus discard (con NIER_SIN_BLEND, el alfa que calculan
        // queda escrito tal cual en el render target).
        if (const char* keep = getenv("NIER_SIN_DESCARTE"); keep && *keep && out.stage == Stage::Fragment)
            for (Shader::IR::Block* block : program.post_order_blocks)
                for (Shader::IR::Inst& inst : block->Instructions())
                    if (inst.GetOpcode() == Shader::IR::Opcode::DemoteToHelperInvocation)
                        inst.ReplaceOpcode(Shader::IR::Opcode::Void);

        out.has_brx = env.HasBrx();

        if (out.stage == Stage::Geometry) {
            // Lecturas que no existen en un geometry shader (la cabecera las marca en algunos de NieR) y
            // que Eden declararía como entradas inválidas: se leerán como 0.
            using A = Shader::IR::Attribute;
            for (A a : {A::InstanceId, A::VertexId, A::FrontFace, A::PointSpriteS, A::PointSpriteT,
                        A::TessellationEvaluationPointU, A::TessellationEvaluationPointV, A::Layer, A::ViewportIndex})
                program.info.loads.Set(a, false), program.info.passthrough.Set(a, false);
        }
        const Shader::Info& info = program.info;
        out.constant_buffer_mask = info.constant_buffer_mask;
        for (size_t k = 0; k < 18 && k < info.constant_buffer_used_sizes.size(); k++)
            out.constant_buffer_used[k] = info.constant_buffer_used_sizes[k];
        for (size_t k = 0; k < 32; k++) {
            if (info.loads.Generic(k)) out.input_generics |= uint64_t{1} << k;
            if (info.stores.Generic(k)) out.output_generics |= uint64_t{1} << k;
        }
        for (size_t k = 0; k < info.stores_frag_color.size(); k++)
            if (info.stores_frag_color[k]) out.color_outputs |= 1u << k;
        const uint32_t base = binding_base(out.stage);
        list_resources(info, base, out);

        Shader::Profile profile{};
        profile.supported_spirv = options.spirv_version;
        profile.unified_descriptor_binding = true;
        profile.support_multi_viewport = true;  // el Adreno lo tiene (y el dispositivo activa todas sus features)
        // Subgrupos: votos y ballot solo si el dispositivo los tiene, y en las etapas que los admite; si el
        // subgrupo es mayor que el warp de 32 de Maxwell, Eden emula los votos con ballot.
        constexpr uint32_t kVote = 0x2, kBallot = 0x8, kShuffle = 0x10, kShuffleRelative = 0x20, kQuad = 0x80;
        const uint32_t ops = options.subgroup_operations;
        const bool usable = (ops & kVote) && (ops & kBallot) && (ops & kShuffle);
        profile.support_vote = usable;
        profile.support_shuffle_relative = (ops & kShuffleRelative) != 0;
        profile.support_quad_shuffles = (ops & kQuad) != 0;
        profile.warp_size_potentially_larger_than_guest = options.subgroup_size > 32;
        profile.max_subgroup_size = options.subgroup_size;
        profile.has_broken_spirv_subgroup_shuffle = options.broken_subgroup_shuffle;
        profile.has_broken_spirv_unreachable = options.broken_unreachable;
        profile.support_sampled_image_array_nonuniform_indexing = options.nonuniform_indexing;
        profile.supported_subgroup_stages = 0;
        if (usable) {
            static const uint32_t kVkStage[] = {0x1, 0x2, 0x4, 0x8, 0x10, 0x20};  // VertexB … Compute
            for (uint32_t k = 0; k < 6; k++)
                if (options.subgroup_stages & kVkStage[k]) profile.supported_subgroup_stages |= 1u << k;
        }
        Shader::RuntimeInfo runtime{};
        // NVN enlaza las etapas en el momento del draw (un programa por etapa): se asume que la etapa
        // anterior escribe todos los atributos genéricos, o Eden no declararía las entradas.
        // Solo genéricos y posición: con todo marcado, un geometry shader arrastraba como entradas valores
        // de sistema que no existen en su etapa (gl_InstanceIndex, gl_TessCoord…) y el driver lo rechazaba.
        for (size_t k = 0; k < 32; k++)
            if (options.previous_stores >> k & 1)
                for (size_t c = 0; c < 4; c++)
                    runtime.previous_stage_stores.Set(Shader::IR::Attribute::Generic0X + k * 4 + c);
        for (size_t c = 0; c < 4; c++) runtime.previous_stage_stores.Set(Shader::IR::Attribute::PositionX + c);
        runtime.input_topology = static_cast<Shader::InputTopology>(options.input_topology);
        runtime.convert_depth_mode = options.convert_depth_mode;
        Shader::Backend::Bindings bindings{};
        bindings.unified = base;
        out.spirv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime, program, bindings);
        move_arrays_to_bindless_set(out);
        if (options.optimize_stages >> static_cast<uint32_t>(out.stage) & 1) {
            const auto start = std::chrono::steady_clock::now();
            out.spirv_opt_from = static_cast<uint32_t>(out.spirv.size());
            out.spirv_opt = optimize_spirv(out.spirv, out.spirv_opt_error) ? 1 : -1;
            out.spirv_opt_us = static_cast<uint32_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
        }
        return true;
    } catch (const std::exception& e) {
        out.error = e.what();
        return false;
    }
}

}  // namespace nvnshader
