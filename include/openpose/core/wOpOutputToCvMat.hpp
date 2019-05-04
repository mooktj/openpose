#ifndef OPENPOSE_CORE_W_OP_OUTPUT_TO_CV_MAT_HPP
#define OPENPOSE_CORE_W_OP_OUTPUT_TO_CV_MAT_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/opOutputToCvMat.hpp>
#include <openpose/thread/worker.hpp>

#include <iostream>
#include <string>

namespace op
{
    template<typename TDatums>
    class WOpOutputToCvMat : public Worker<TDatums>
    {
    public:
        explicit WOpOutputToCvMat(const std::shared_ptr<OpOutputToCvMat>& opOutputToCvMat, const bool vnectEnable);

        virtual ~WOpOutputToCvMat();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<OpOutputToCvMat> spOpOutputToCvMat;
        const bool spVnectEnable;

        DELETE_COPY(WOpOutputToCvMat);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/vnect/vNect.hpp>

namespace op
{
    template<typename TDatums>
    WOpOutputToCvMat<TDatums>::WOpOutputToCvMat(const std::shared_ptr<OpOutputToCvMat>& opOutputToCvMat, const bool vnectEnable) :
        spOpOutputToCvMat{opOutputToCvMat},
        spVnectEnable{vnectEnable}
    {
    }

    template<typename TDatums>
    WOpOutputToCvMat<TDatums>::~WOpOutputToCvMat()
    {
    }

    template<typename TDatums>
    void WOpOutputToCvMat<TDatums>::initializationOnThread()
    {
    }

    template<typename TDatums>
    void WOpOutputToCvMat<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // std::cout << "~~wOpOutputToCvMat\n";
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // float* -> cv::Mat
                for (auto& tDatumPtr : *tDatums)
                {   
                    // auto& poseKeypoints = tDatumPtr->poseKeypoints;

                    // std::cout << "wOpOutputToCvMat BEFORE: tDatumPtr->cvOutputData is empty? " << tDatumPtr->cvOutputData.empty() << "\n";
                    // std::cout << "  wOpOutputToCvMat:: poseKeypoints.getSize(1): " << poseKeypoints.getSize(1) << ", getSize(2): " << poseKeypoints.getSize(2) << ", getSize(3): " << poseKeypoints.getSize(3) << "\n";
                    // std::string pathToWrite = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-1-5-19/";
                    // std::string vname = "media_2_croppedImg_" + std::to_string(2);
                    tDatumPtr->cvOutputData = spOpOutputToCvMat->formatToCvMat(tDatumPtr->outputData);
                    
                    // ------- GET VNECT WORKING (NEED TO MIGRATE TO VNECT OWN CLASS) --------//
                    // if(tDatumPtr->renderVNect) {
                    if(spVnectEnable) {
                        std::cout << "wOpOutputToCvMat:: spVnectEnable!\n";
                        vNectPostForward(tDatumPtr);   
                    }
                    // }
                    // write3dJointsToFile(tDatumPtr, vname, pathToWrite);
                    // std::cout << "wOpOutputToCvMat AFTER: tDatumPtr->cvOutputData is empty? " << tDatumPtr->cvOutputData.empty() << "\n";
                    // std::cout << "  wOpOutputToCvMat:: poseKeypoints.getSize(1): " << poseKeypoints.getSize(1) << ", getSize(2): " << poseKeypoints.getSize(2) << ", getSize(3): " << poseKeypoints.getSize(3) << "\n";
                }
               
                
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WOpOutputToCvMat);
}

#endif // OPENPOSE_CORE_W_OP_OUTPUT_TO_CV_MAT_HPP
