#include <openpose/wrapper/wrapperStructGui.hpp>
#include <iostream>

namespace op
{
    WrapperStructGui::WrapperStructGui(
        const DisplayMode displayMode_, const bool guiVerbose_, const bool fullScreen_) :
        displayMode{displayMode_},
        guiVerbose{guiVerbose_},
        fullScreen{fullScreen_}
    {
    	// std::cout << "wrapperStructGui:: WrapperStructGui(...) constructor\n";
    }
}
