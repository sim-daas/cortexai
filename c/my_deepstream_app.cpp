#include "pipeline.hpp"
#include <iostream>

using namespace deepstream;
#define CONFIG_FILE_PATH "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_infer_primary.yml"

int main(int argc, char *argv[])
{
    try {
        Pipeline pipeline("sample-pipeline");
        pipeline.add("nvurisrcbin", "src", "uri", argv[1])
                .add("nvstreammux", "mux", "batch-size", 1, "width", 1280, "height", 720)
                .add("nvinferbin", "infer", "config-file-path", CONFIG_FILE_PATH)
                .add("nvosdbin", "osd")
                .add("nveglglessink", "sink")
                .link({"src", "mux"}, {"", "sink_%u"}).link("mux","infer", "osd", "sink");
        pipeline.start().wait();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}