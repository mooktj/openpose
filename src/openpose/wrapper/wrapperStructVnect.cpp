#include <openpose/wrapper/wrapperStructVnect.hpp>
#include <iostream>

namespace op
{
    WrapperStructVnect::WrapperStructVnect(
        const bool vnectEnable_, const std::string& modelFolder_,
        const std::string& protoTxtFile_, const std::string& trainedModelFile_) :
        vnectEnable{vnectEnable_},
        modelFolder{modelFolder_},
        protoTxtFile{protoTxtFile_},
        trainedModelFile{trainedModelFile_}
    {
        std::cout << "wrapperStructVnect:: WrapperStructVnect(...) constructor\n";
    }
}
