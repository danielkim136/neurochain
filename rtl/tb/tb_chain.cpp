// Verilator testbench for nc_sbp_chain: plays ADC codes from a file through
// the model and writes every output to files, which Python then compares with
// the integer reference (neurochain.rtl.spec.ChainSpec.reference).
//
// The same file drives the RTL and the gate-level netlist Yosys writes: the two
// have the same ports.
//
// Usage: tb_chain <codes.bin> <trig.bin> <out_prefix> [activity.saif]
//   codes.bin  int8, n_samples rows of TB_N_CH codes (channel is the fast index)
//   trig.bin   uint8, one flag per sample: a stimulation pulse starts here
// Writes <out_prefix>.c.bin and .y.bin (uint16, raw bits of the signed taps),
// .ok.bin (uint8), .sum.bin (uint32) and .cnt.bin (uint8), in the order the
// samples went in, and prints one line of JSON with the counts.
//
// Compile-time: -DTB_N_CH=<channels> -DTB_CODE_W=<ADC bits>. A SAIF activity
// file is written only by a model built with --trace-saif.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#include "Vnc_sbp_chain.h"
#include "verilated.h"
#if VM_TRACE
#include "verilated_saif_c.h"
#endif

#if !defined(TB_N_CH) || !defined(TB_CODE_W)
#error "compile with -DTB_N_CH=<number of channels> -DTB_CODE_W=<ADC bits>"
#endif

namespace {

std::vector<uint8_t> read_file(const char* path) {
    std::FILE* f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "tb_chain: cannot open %s\n", path);
        std::exit(2);
    }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(static_cast<size_t>(size));
    if (size > 0 && std::fread(data.data(), 1, data.size(), f) != data.size()) {
        std::fprintf(stderr, "tb_chain: short read on %s\n", path);
        std::exit(2);
    }
    std::fclose(f);
    return data;
}

std::FILE* open_out(const std::string& prefix, const char* suffix) {
    const std::string path = prefix + suffix;
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "tb_chain: cannot write %s\n", path.c_str());
        std::exit(2);
    }
    return f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s codes.bin trig.bin out_prefix [activity.saif]\n", argv[0]);
        return 2;
    }
    const std::vector<uint8_t> codes = read_file(argv[1]);
    const std::vector<uint8_t> trig = read_file(argv[2]);
    const std::string prefix = argv[3];
    const size_t n_ch = TB_N_CH;
    const size_t n_samples = trig.size();
    if (codes.size() != n_samples * n_ch) {
        std::fprintf(stderr, "tb_chain: %zu codes for %zu samples of %zu channels\n", codes.size(),
                     n_samples, n_ch);
        return 2;
    }

    const std::unique_ptr<VerilatedContext> context{new VerilatedContext};
    context->commandArgs(argc, argv);
    const std::unique_ptr<Vnc_sbp_chain> top{new Vnc_sbp_chain{context.get()}};
#if VM_TRACE
    std::unique_ptr<VerilatedSaifC> saif;
    if (argc > 4) {
        context->traceEverOn(true);
        saif.reset(new VerilatedSaifC);
        top->trace(saif.get(), 99);
        saif->open(argv[4]);
    }
#endif

    std::FILE* f_c = open_out(prefix, ".c.bin");
    std::FILE* f_y = open_out(prefix, ".y.bin");
    std::FILE* f_ok = open_out(prefix, ".ok.bin");
    std::FILE* f_sum = open_out(prefix, ".sum.bin");
    std::FILE* f_cnt = open_out(prefix, ".cnt.bin");

    uint64_t cycles = 0, n_smp = 0, n_bin = 0, tag_errors = 0, active_cycles = 0;
    bool counting = false;

    // One clock cycle: rising edge, look at the outputs, falling edge.
    auto tick = [&]() {
        top->clk = 1;
        top->eval();
#if VM_TRACE
        if (saif && counting) saif->dump(2 * cycles);
#endif
        if (top->smp_valid) {
            if (top->smp_ch != n_smp % n_ch) ++tag_errors;
            const uint16_t c = top->smp_c, y = top->smp_y;
            const uint8_t ok = top->smp_ok;
            std::fwrite(&c, sizeof c, 1, f_c);
            std::fwrite(&y, sizeof y, 1, f_y);
            std::fwrite(&ok, sizeof ok, 1, f_ok);
            ++n_smp;
        }
        if (top->bin_valid) {
            if (top->bin_ch != n_bin % n_ch) ++tag_errors;
            const uint32_t sum = top->bin_sum;
            const uint8_t cnt = top->bin_cnt;
            std::fwrite(&sum, sizeof sum, 1, f_sum);
            std::fwrite(&cnt, sizeof cnt, 1, f_cnt);
            ++n_bin;
        }
        top->clk = 0;
        top->eval();
#if VM_TRACE
        if (saif && counting) saif->dump(2 * cycles + 1);
#endif
        ++cycles;
        if (counting) ++active_cycles;
    };

    const auto start = std::chrono::steady_clock::now();
    top->clk = 0;
    top->rst = 1;
    top->in_valid = 0;
    top->in_ch = 0;
    top->in_code = 0;
    top->in_trig = 0;
    for (int i = 0; i < 4; ++i) tick();
    top->rst = 0;
    while (!top->ready) tick();

    // Activity is counted over the cycles that carry samples, not the reset.
    counting = true;
    for (size_t n = 0; n < n_samples; ++n) {
        const uint8_t* row = &codes[n * n_ch];
        for (size_t ch = 0; ch < n_ch; ++ch) {
            top->in_valid = 1;
            top->in_ch = ch;
            // Verilator does not mask a port wider than its signal: keep CODE_W bits.
            top->in_code = row[ch] & ((1u << TB_CODE_W) - 1u);
            top->in_trig = trig[n];
            tick();
            if (n_ch < 2) {  // the spacing rule: never the same channel twice in a row
                top->in_valid = 0;
                tick();
            }
        }
    }
    top->in_valid = 0;
    for (int i = 0; i < 8; ++i) tick();
    counting = false;
    const double wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

#if VM_TRACE
    if (saif) saif->close();
#endif
    top->final();
    for (std::FILE* f : {f_c, f_y, f_ok, f_sum, f_cnt}) std::fclose(f);
    std::printf(
        "{\"n_channels\": %zu, \"n_samples\": %zu, \"cycles\": %llu, \"active_cycles\": %llu, "
        "\"samples_out\": %llu, \"bins_out\": %llu, \"tag_errors\": %llu, \"wall_s\": %.3f}\n",
        n_ch, n_samples, static_cast<unsigned long long>(cycles),
        static_cast<unsigned long long>(active_cycles), static_cast<unsigned long long>(n_smp),
        static_cast<unsigned long long>(n_bin), static_cast<unsigned long long>(tag_errors),
        wall_s);
    return 0;
}
