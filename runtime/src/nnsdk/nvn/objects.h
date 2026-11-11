#pragma once
#include <atomic>
// Objetos NVN inicializados que comparten varios archivos (pools, texturas, samplers, programas,
// ventana). Los objetos del juego son opacos: se asocian por dirección a estos.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "nnsdk/nvn/gpu_memory.h"
#include "nnsdk/nvn/layout.h"
#include "nnsdk/nvn/nvn.h"
#include "nnsdk/nvn/spirv_cache.h"
#include "nnsdk/nvn/vk.h"
#include "nvn_shader.h"

namespace nvn {

struct MemoryPool {
    uint8_t* storage = nullptr;  // memoria del juego; también es la "dirección de GPU"
    size_t size = 0;
    int flags = 0;
    // Pool virtual (memory.cpp): sin memoria propia; storage es una reserva nuestra donde MapVirtual hace aparecer las
    // páginas de pools físicos (lo no mapeado se lee como ceros).
    bool reserved = false;
};
// NVNmemoryPoolFlags que cambian lo que hacemos (el resto son de caché y acceso).
constexpr int kPoolFlagPhysical = 0x100, kPoolFlagVirtual = 0x200;

// Estado de un NVNtextureBuilder; vive dentro del builder del juego (≤ 96 bytes).
struct TextureDesc {
    void* device;
    int target, flags, format;
    int width, height, depth, levels, samples;
    int stride;
    void* pool;
    int64_t offset;
    uint8_t depth_stencil_mode;  // NVNtextureDepthStencilMode: DEPTH 0, STENCIL 1 (lo que se muestrea)
    uint8_t swizzle[4];          // NVNtextureSwizzle (ZERO 0, ONE 1, R 2 … A 5); SetDefaults: R G B A
    // SetPackagedTextureLayout: la disposición con la que el empaquetador de texturas dejó los datos (8 bytes de NVN).
    uint8_t has_packaged_layout;
    uint8_t packaged_layout[8];
};
static_assert(sizeof(TextureDesc) <= 96);

constexpr int kTextureFlagDisplay = 0x1, kTextureFlagImage = 0x4, kTextureFlagLinear = 0x10;
constexpr int kTarget1D = 0, kTarget2D = 1, kTarget3D = 2, kTarget1DArray = 3, kTarget2DArray = 4,
              kTarget2DMultisample = 5, kTarget2DMultisampleArray = 6, kTargetRectangle = 7, kTargetCube = 8,
              kTargetCubeArray = 9, kTargetBuffer = 10;

// NVNtextureView: vive dentro del struct del juego (≤ 40 bytes). flags dice qué campos valen.
struct TextureViewDesc {
    uint16_t flags;
    uint8_t depth_stencil_mode, pad;
    int16_t base_level, levels;
    int32_t base_layer, layers;
    int32_t format, target;
    uint8_t swizzle[4];
};
static_assert(sizeof(TextureViewDesc) <= 40);
constexpr uint16_t kViewLevels = 1, kViewLayers = 2, kViewFormat = 4, kViewTarget = 8, kViewDepthStencil = 16,
                   kViewSwizzle = 32;

struct Texture {
    // Único en toda la ejecución (nunca se reutiliza, al contrario que la dirección del objeto): clave de
    // las cachés de vistas. Cambia al reinicializar o finalizar la textura.
    uint64_t serial = 0;
    TextureDesc desc{};
    TextureLayout layout;
    uint8_t* storage = nullptr;  // memoria del juego donde están sus texels (si los sube la CPU)
    // Vulkan, creado cuando hace falta:
    VkImage image = VK_NULL_HANDLE;
    GpuAllocation memory;  // dentro de un bloque compartido o propia (gpu_memory.h)
    uint64_t gpu_bytes = 0;
    VkFormat vk_format = VK_FORMAT_UNDEFINED;
    VkImageAspectFlags aspect = 0;
    uint32_t layers = 1;
    bool vk_failed = false;    // no se pudo crear en Vulkan (formato sin soporte…): no reintentar
    bool mutable_format = true;  // imagen con MUTABLE_FORMAT: admite vistas con otro formato (ejecutor)
    uint64_t retry_frame = 0;  // sin memoria al crearla: no se reintenta antes de este frame
    bool gpu_written = false;  // ha sido render target o destino de una copia: no se sube de la CPU
    // Último frame del ejecutor en que fue render target, y si también lo fue en uno de los dos anteriores (se redibuja
    // cada frame: un draw que no se dibuje uno de ellos sale en el siguiente; ver las pipelines en segundo plano).
    uint64_t rt_frame = 0;
    bool rt_recurring = false;
    bool storage_image = false;  // la imagen se creó con STORAGE: se puede enlazar como imagen (ejecutor)
    bool uploaded = false;
    uint64_t content_hash = 0;   // de su memoria al subirla: si cambia, el juego la ha reescrito
    uint64_t checked_frame = ~uint64_t{0};
    float clear_color[4] = {};  // último ClearColor (para presentar antes de tener render real)
    bool cleared = false;
    // Ejecutor: la última vista muestreada (válida si view_memo_serial == serial).
    VkImageView view_memo = VK_NULL_HANDLE;
    // Primer nivel con memoria detrás (texturas sparse por streaming: MH Rise mapea los mips grandes cuando le hacen
    // falta). Lo muestreado empieza ahí: los niveles sin mapear se leen a cero, que en ASTC es el magenta de error.
    int min_level = 0;
    // Huella muestreada de cada nivel (capa x nivel) de la última subida: al cambiar la memoria solo se vuelven a subir
    // los niveles que cambiaron (el streaming va trayendo tiles de los mips grandes frame a frame).
    std::vector<uint64_t> level_hash;
    VkImage level_hash_image = VK_NULL_HANDLE;  // la imagen a la que corresponden (otra imagen: se sube entera)
    // Capa x nivel que la próxima subida copia aunque su huella muestreada no haya cambiado: lo que escribió WriteTexels o
    // CopyBufferToTexture (1) o lo que la vigilancia vio escrito (2; 4: y su huella entera ya dijo que cambió). La huella
    // (64 bytes de cada 4 KB) no ve casi nunca un glifo nuevo en el atlas de la UI de MH Rise (1024x1024 R8, block-linear:
    // 64 bytes = 32x2 texels de cada 64x64): se veía la letra que había antes en ese hueco hasta otra subida.
    std::vector<uint8_t> level_forced;
    // Huella entera (todos los bytes) de cada capa x nivel tal como se subió, 0 si no se sabe: lo que una escritura exacta
    // marca pero quedó igual (el juego reescribe el atlas entero, o páginas con alias) no se vuelve a subir.
    std::vector<uint64_t> level_full_hash;
    // Vigilancia de escrituras de la CPU en su memoria (nnsdk/watch.h, como los espejos): las que sube la CPU sin
    // comprimir (el atlas de glifos) se revisan por páginas en cada envío en vez de por la huella muestreada una vez por
    // frame. Una clave por página de 4 KB; el envío de la última revisión.
    bool watched = false;
    // No se vigila: recibe WriteTexels/CopyBufferToTexture (sus cajas ya dicen qué cambia) o la CPU la reescribe entera una
    // y otra vez (vídeo: saltaría en cada página de cada frame; la huella muestreada ya lo ve).
    bool no_watch = false;
    uint8_t watch_rewrites = 0;  // cambios seguidos con más de media textura escrita
    std::vector<uint64_t> watch_keys;
    uint64_t watch_submit = ~uint64_t{0};
    uint64_t view_memo_serial = 0;
    TextureViewDesc view_memo_desc{};
    // Cajas que la CPU escribió con nvnTextureWriteTexels (texture.cpp) y que la imagen aún no tiene: el ejecutor las
    // sube al usar la textura. Con texel_writes_mutex(); g_texel_writes cuenta las pendientes de todas las texturas.
    struct TexelWrite {
        uint32_t level, layer, layers;
        BlockBox box;
    };
    std::vector<TexelWrite> texel_writes;
};

std::mutex& texel_writes_mutex();
extern std::atomic<uint32_t> g_texel_writes;

// Libera la imagen de Vulkan de una textura (texture.cpp); el objeto sigue siendo válido.
void release_texture_gpu(Texture& t);
uint64_t next_texture_serial();
// Memoria de GPU de las texturas vivas (DEPURACIÓN/perfil).
extern std::atomic<uint64_t> g_texture_gpu_bytes, g_texture_gpu_images;

struct SamplerDesc {  // NVNsamplerBuilder (≤ 96 bytes)
    void* device;
    uint8_t min_filter, mag_filter, wrap[3], compare_mode, compare_func, reduction;
    float max_anisotropy, lod_min, lod_max, lod_bias;
    float border[4];
};
static_assert(sizeof(SamplerDesc) <= 96);

struct Sampler {
    SamplerDesc desc{};
    VkSampler vk = VK_NULL_HANDLE;
    // El otro anisotrópico (ver Executor::sampler), en vivo. En los de comparación (sombras): el de antes, con el mínimo de
    // 4x forzado (<datos>/con_af_comparacion). En los demás: sin ese mínimo, el que pide el juego (<datos>/af_del_juego).
    // Se retira con vk.
    VkSampler vk_af = VK_NULL_HANDLE;
};

struct TexturePoolEntry {
    Texture* texture = nullptr;
    TextureViewDesc view{};
};

struct DescriptorPool {
    void* memory_pool = nullptr;
    int64_t offset = 0;
    int count = 0;
    std::vector<TexturePoolEntry> textures;  // pool de texturas: id -> textura + vista
    std::vector<Sampler*> samplers;          // pool de samplers: id -> sampler
};

struct ProgramStage {
    // Geometry shader de la compactación de NieR (pre-pase de profundidad): escribe la lista de triángulos visibles y sus
    // argumentos indirectos, que ya no usamos (el pase de color dibuja la malla entera, ver culled_draws_).
    bool compaction_gs = false;
    uint64_t code = 0;  // dirección del código (NVN solo lo lee al dibujar: se traduce en el executor)
    uint64_t print = 0;  // huella del código (la de los logs), al traducirlo
    const void* control = nullptr;  // sección de control (memoria de CPU del juego)
    // Constantes del propio shader (su c1): el compilador las deja tras el código, en
    // code + control[0x700] hasta code + control[0x704].
    uint64_t constants = 0;
    uint32_t constants_size = 0;
    bool tried = false;
    nvnshader::Translated shader;
    VkShaderModule module = VK_NULL_HANDLE;
    // Compute de kUniformBarrierShaders (program.cpp), a los que se les dejan las barreras de sus bucles: la traducción
    // sin ellas, para sus interruptores en vivo (use_stripped_module). Los mismos recursos que module (el mismo layout).
    VkShaderModule stripped_module = VK_NULL_HANDLE;
    int8_t uniform_barrier_entry = -1;  // su índice en kUniformBarrierShaders (-1: no es de la lista)
    // Identidad de lo traducido (spirv_identity: el SPIR-V y lo que el layout toma de los recursos). Las pipelines y los
    // layouts del ejecutor van por ella y no por la dirección de la etapa: el juego crea varios programas con el mismo
    // código (MH Rise: 2200 etapas de 496 huellas en una partida) y cada uno llevaba sus propias pipelines. 0: sin traducir.
    uint64_t spirv_id = 0;
    // Clave con la que está en la caché de SPIR-V (program.cpp); a cero si no está (sin sección de control, caché
    // apagada). El ejecutor la apunta con cada pipeline nueva para crearla al arrancar la partida siguiente (precalentado).
    spirv_cache::Key cache_key;
    // Variantes: geometry por (genéricos que escribe el vertex shader, topología de entrada); pixel por los genéricos que
    // le llegan (los que no escribe la etapa anterior no se declaran).
    struct Variant {
        VkShaderModule module = VK_NULL_HANDLE;
        uint64_t spirv_id = 0;
        spirv_cache::Key cache_key;
    };
    std::unordered_map<uint64_t, Variant> variants;
};

struct Program {
    std::vector<ProgramStage> stages;
    uint64_t generation = 0;
};

struct Window {
    void* native_window = nullptr;
    int present_interval = 1;
    std::vector<void*> textures;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkExtent2D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;  // el del swapchain (el HUD lo necesita)
    std::vector<VkImage> images;
    int next_texture = 0;
    uint64_t presented = 0;
    // nvnWindowSetCrop: la parte de la textura que va a la pantalla (ancho 0: entera). MH Rise dibuja a 1280x720 en
    // texturas de 1920x1080 y recorta.
    VkRect2D crop{};
};

extern Objects<MemoryPool> g_memory_pools;
extern Objects<Texture> g_textures;
extern Objects<Window> g_windows;
extern Objects<DescriptorPool> g_texture_pools, g_sampler_pools;

Program* find_program(const void* program);
// Traduce la etapa la primera vez que se usa (program.cpp); false si no se pudo.
bool ensure_translated(ProgramStage& stage);
// Precarga (program.cpp): el hilo que traduce los shaders al fijarlos le pasa al ejecutor cada compute traducido, para
// crear su pipeline antes del primer dispatch (la toma warm_take).
void prefetch_compute_pipeline(const ProgramStage& stage);

// Lo que pasó en el último tramo del log de ritmo (window.cpp lo escribe cada 10 s si hubo frames lentos y lo pone a
// cero): GPU del frame del juego, el ejecutor por frame, pipelines y traducciones en NvnSubmit y en segundo plano.
struct DropStats {
    std::atomic<uint64_t> gpu_sum_ns{0}, gpu_max_ns{0}, gpu_frames{0};
    std::atomic<uint64_t> run_ns{0}, run_max_ns{0}, wait_ns{0};  // run_max: el peor frame (suma de sus envíos)
    std::atomic<uint64_t> pipelines_live{0}, pipelines_live_ns{0};   // creadas en NvnSubmit con el frame parado
    std::atomic<uint64_t> pipelines_async{0}, draws_skipped{0};     // mandadas al hilo y draws sin dibujar esperándolas
    std::atomic<uint64_t> translated_live{0}, translated_live_ns{0};  // traducciones (no de la caché) en NvnSubmit
    std::atomic<uint64_t> prefetched{0}, prefetch_ns{0};             // las del hilo de la precarga
    std::mutex slow_mutex;      // el envío más lento del tramo (de 40 ms o más) y en qué se le fue el tiempo
    uint64_t slow_ns = 0;
    std::string slow_summary;
};
extern DropStats g_drop_stats;
inline void note_max(std::atomic<uint64_t>& a, uint64_t v) {
    uint64_t now = a.load(std::memory_order_relaxed);
    while (v > now && !a.compare_exchange_weak(now, v, std::memory_order_relaxed)) {
    }
}
// true si hay que usar stripped_module en vez de module: un interruptor en vivo devuelve ese compute a la traducción
// sin las barreras de sus bucles (<datos>/sin_barrera_exposicion, sin_barrera_reduccion o sin_barrera_uniforme: todos).
bool use_stripped_module(const ProgramStage& stage);
// Geometry: el módulo para un vertex shader que escribe `previous_stores` y una topología de entrada de
// Eden; module a VK_NULL_HANDLE si no se puede traducir.
ProgramStage::Variant geometry_variant(ProgramStage& stage, uint64_t previous_stores, int input_topology);
// Identidad de una traducción (ProgramStage::spirv_id): huella del SPIR-V, de la etapa, de los genéricos y de lo que el
// layout toma de cada recurso (tipo, set, binding, número). Dos etapas con la misma comparten pipelines y layouts.
uint64_t spirv_identity(const nvnshader::Translated& t);
// Precalentado (ejecutor): espera a que la caché de SPIR-V esté leída (la abre si hacía falta); false si está apagada
// o no llegó a tiempo. cached_translation: la traducción guardada con esa clave.
bool shader_cache_ready(int timeout_ms);
bool cached_translation(const spirv_cache::Key& key, nvnshader::Translated& out);
// DEPURACIÓN: guarda el código de la etapa en <datos>/shaders/<name>.code.
void dump_stage_code(const ProgramStage& stage, const std::string& name);
struct CopyRegion;
// Como nvnTextureWriteTexels: texels lineales (compactos si los strides son 0) a la memoria de la textura, en su
// disposición; la caja se sube a la imagen al usarla. Para CopyBufferToTexture, que en la Switch copia la GPU.
void write_texels_to(Texture& t, const TextureViewDesc* view, const CopyRegion* region, const void* data, int64_t row_stride,
                     int64_t image_stride);
Sampler* find_sampler(const void* sampler);

// Handles de NVN (nuestro formato): separados = id; combinado = sampler << 32 | textura.
inline uint32_t handle_texture_id(uint64_t h) { return static_cast<uint32_t>(h & 0xFFFFFFFF); }
inline uint32_t handle_sampler_id(uint64_t h) { return static_cast<uint32_t>(h >> 32); }

// Presenta en pantalla la textura index de la ventana (window.cpp).
void present(Window& window, int index);

}  // namespace nvn
