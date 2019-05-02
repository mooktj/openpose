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
        explicit WOpOutputToCvMat(const std::shared_ptr<OpOutputToCvMat>& opOutputToCvMat);

        virtual ~WOpOutputToCvMat();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        const std::shared_ptr<OpOutputToCvMat> spOpOutputToCvMat;

        DELETE_COPY(WOpOutputToCvMat);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/vnect/vNect.hpp>

namespace op
{
    template<typename TDatums>
    WOpOutputToCvMat<TDatums>::WOpOutputToCvMat(const std::shared_ptr<OpOutputToCvMat>& opOutputToCvMat) :
        spOpOutputToCvMat{opOutputToCvMat}
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
                std::cout << "~~wOpOutputToCvMat\n";
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // float* -> cv::Mat
                for (auto& tDatumPtr : *tDatums)
                {   
                    auto& poseKeypoints = tDatumPtr->poseKeypoints;

                    // std::cout << "wOpOutputToCvMat BEFORE: tDatumPtr->cvOutputData is empty? " << tDatumPtr->cvOutputData.empty() << "\n";
                    // std::cout << "  wOpOutputToCvMat:: poseKeypoints.getSize(1): " << poseKeypoints.getSize(1) << ", getSize(2): " << poseKeypoints.getSize(2) << ", getSize(3): " << poseKeypoints.getSize(3) << "\n";
                   
                    tDatumPtr->cvOutputData = spOpOutputToCvMat->formatToCvMat(tDatumPtr->outputData);
                    

                    vNectPostForward(tDatumPtr);
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
