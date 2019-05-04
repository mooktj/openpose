#ifndef OPENPOSE_WRAPPER_WRAPPER_STRUCT_VNECT_HPP
#define OPENPOSE_WRAPPER_WRAPPER_STRUCT_VNECT_HPP

#include <openpose/core/common.hpp>
#include <openpose/filestream/enumClasses.hpp>

namespace op
{
    /**
     * WrapperStructVnect: Output ( writing rendered results and/or pose data, etc.) configuration struct.
     */
    struct OP_API WrapperStructVnect
    {
        
        bool vnectEnable;

        // float[] dataScale;

        std::string modelFolder;

        std::string protoTxtFile;

        std::string trainedModelFile;
        /**
         * Constructor of the struct.
         * It has the recommended and default values we recommend for each element of the struct.
         * Since all the elements of the struct are public, they can also be manually filled.
         */
        WrapperStructVnect(
            const bool vnectEnable = false, const std::string& modelFolder = "",
            const std::string& protoTxtFile = "", const std::string& trainedModelFile = "");
    };
}

#endif // OPENPOSE_WRAPPER_WRAPPER_STRUCT_VNECT_HPP
