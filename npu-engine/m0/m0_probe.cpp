// M0 probe: prove our own process can reach the NPU via XRT, load an FLM xclbin,
// and enumerate its kernels. First reachability step of the open engine — no FLM
// engine involved, just XRT + one of FLM's kernel binaries.
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/experimental/xrt_xclbin.h"
#include <cstdio>
#include <exception>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { printf("usage: m0_probe <path-to.xclbin>\n"); return 2; }
    try {
        xrt::device device(0);
        try {
            printf("device[0]: %s\n", device.get_info<xrt::info::device::name>().c_str());
        } catch (...) { printf("device[0] opened (name query unsupported)\n"); }

        std::string xclbin_path(argv[1]);
        xrt::xclbin xcl{xclbin_path};
        printf("xclbin loaded: %s\n", argv[1]);

        auto uuid = device.register_xclbin(xcl);
        printf("registered xclbin, uuid=%s\n", uuid.to_string().c_str());

        xrt::hw_context ctx(device, uuid);
        printf("hw_context created OK\n");

        std::vector<xrt::xclbin::kernel> kernel_list = xcl.get_kernels();
        printf("kernels: %zu\n", kernel_list.size());
        for (size_t kidx = 0; kidx < kernel_list.size(); ++kidx) {
            const xrt::xclbin::kernel& kern = kernel_list[kidx];
            printf("  kernel '%s' nargs=%u\n", kern.get_name().c_str(), (unsigned)kern.get_num_args());
            std::vector<xrt::xclbin::arg> arg_list = kern.get_args();
            for (size_t aidx = 0; aidx < arg_list.size(); ++aidx)
                printf("    arg[%d] '%s'\n", (int)arg_list[aidx].get_index(),
                       arg_list[aidx].get_name().c_str());
        }
        printf("M0-STEP1 OK: device + xclbin + hw_context + kernel enumeration\n");
    } catch (const std::exception& e) {
        printf("EXCEPTION: %s\n", e.what());
        return 1;
    }
    return 0;
}
