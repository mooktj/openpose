#ifndef OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP
#define OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/poseExtractor.hpp>
#include <openpose/thread/worker.hpp>
#include <iostream>

namespace op
{
    template<typename TDatums>
    class WPoseExtractor : public Worker<TDatums>
    {
    public:
        explicit WPoseExtractor(const std::shared_ptr<PoseExtractor>& poseExtractorSharedPtr);

        virtual ~WPoseExtractor();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<PoseExtractor> spPoseExtractor;

        DELETE_COPY(WPoseExtractor);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
namespace op
{
    template<typename TDatums>
    WPoseExtractor<TDatums>::WPoseExtractor(const std::shared_ptr<PoseExtractor>& poseExtractorSharedPtr) :
        spPoseExtractor{poseExtractorSharedPtr}
    {
        // std::cout << "wPoseExtractor:: WPoseExtractor(...) constructor\n";
    }

    template<typename TDatums>
    WPoseExtractor<TDatums>::~WPoseExtractor()
    {
        // std::cout << "wPoseExtractor:: ~WPoseExtractor()\n";
    }

    template<typename TDatums>
    void WPoseExtractor<TDatums>::initializationOnThread()
    {
        // std::cout << "wPoseExtractor:: initializationOnThread()\n";
        try
        {
            spPoseExtractor->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template<typename TDatums>
    void WPoseExtractor<TDatums>::work(TDatums& tDatums)
    {
        // std::cout << "wPoseExtractor:: work(tDatums)\n"; //outputs a lot --> seems to be calling all the time
        try
        {
            if (checkNoNullNorEmpty(tDatums))
            {
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);

                // std::cout << "---->wPoseExtractor:: tDatums is not Null or Empty\n";
                // Extract people pose
                for (auto i = 0u ; i < tDatums->size() ; i++)
                // for (auto& tDatum : *tDatums)
                {
                    // std::cout << "------>wPoseExtractor:: going to forwardPass to spPoseExtractor\n";
                    auto& tDatumPtr = (*tDatums)[i];
                    // OpenPose net forward pass
                    spPoseExtractor->forwardPass(
                        tDatumPtr->inputNetData, Point<int>{tDatumPtr->cvInputData.cols, tDatumPtr->cvInputData.rows},
                        tDatumPtr->scaleInputToNetInputs, tDatumPtr->poseNetOutput, tDatumPtr->id);
                    // std::cout << "------>wPoseExtractor:: went to forwardPass\n";
                    // OpenPose keypoint detector
                    // std::cout << "------>wPoseExtractor:: spPoseExtractor->getCandidatesCopy()\n";
                    // std::cout << "------>wPoseExtractor:: spPoseExtractor->getHeatMapsCopy()\n";
                    tDatumPtr->poseCandidates = spPoseExtractor->getCandidatesCopy();
                    // std::cout << "size of tDatumPtr->poseScores: " << sizeof(tDatumPtr->poseScores)/sizeof((tDatumPtr->poseScores)[0]) << "\n";
                    // std::cout << tDatumPtr->poseCandidates[0].size() << "\n";

                    // std::cout << "--\n";
                    // std::cout << "--\n";
                    // std::cout << "--\n";
                    // std::cout << "--\n";

                    // std::cout << "wPoseExtractor:: going to getHeatMapsCopy()\n";


                    tDatumPtr->poseHeatMaps = spPoseExtractor->getHeatMapsCopy();
                    // std::cout << "********^^Mookie the Rookie Cookie Smoothie^^presents wPoseExtractor tDatumPtr->poseHeatMaps.size(): " << sizeof(tDatumPtr->poseHeatMaps) << "\n";
                    
                    // std::cout << tDatumPtr->poseHeatMaps << "\n";
                    // std::cout << "--\n";
                    // std::cout << "--\n";
                    // std::cout << "--\n";
                    // std::cout << "--\n";
                    tDatumPtr->poseKeypoints = spPoseExtractor->getPoseKeypoints().clone();
                    tDatumPtr->poseScores = spPoseExtractor->getPoseScores().clone();
                    tDatumPtr->scaleNetToOutput = spPoseExtractor->getScaleNetToOutput();

                     // std::cout << "size of tDatumPtr->poseScores: " << sizeof(tDatumPtr->poseScores)/sizeof((tDatumPtr->poseScores)[0]) << "\n";
                    // Keep desired top N people
                    spPoseExtractor->keepTopPeople(tDatumPtr->poseKeypoints, tDatumPtr->poseScores);
                    // ID extractor (experimental)
                    tDatumPtr->poseIds = spPoseExtractor->extractIdsLockThread(
                        tDatumPtr->poseKeypoints, tDatumPtr->cvInputData, i, tDatumPtr->id);
                    // Tracking (experimental)
                    spPoseExtractor->trackLockThread(
                        tDatumPtr->poseKeypoints, tDatumPtr->poseIds, tDatumPtr->cvInputData, i, tDatumPtr->id);
                }
                // Profiling speed
                Profiler::timerEnd(profilerKey);
                Profiler::printAveragedTimeMsOnIterationX(profilerKey, __LINE__, __FUNCTION__, __FILE__);
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);

                /////////////////////////////////////////////////////////////////////////////////////////
                ////////////////////PRINT DATUM INFO/////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////////

                    // std::cout << "//////////////////PRINT DATUM INFO//////////////////\n";
                    // std::cout << "size: " << tDatums->size() << "\n";
                    // std::cout << "id: " << tDatums->at(0)->id << "\n";
                    // std::cout << "cvInputData size: " << tDatums->at(0)->cvInputData.size() << "\n";
                    // std::cout << "inputNetData size: " << tDatums->at(0)->inputNetData.size() << "\n";
                    // std::cout << "outputData size: " << sizeof(tDatums->at(0)->outputData)/sizeof(tDatums->at(0)->outputData[0]) << "\n";
                    // std::cout << "cvOutputData size: " << tDatums->at(0)->cvOutputData << "\n";
                    // std::cout << "cvOutputData3D size: " << tDatums->at(0)->cvOutputData3D.size() << "\n";

                    // std::cout << "poseKeypoints size: " << sizeof(tDatums->at(0)->poseKeypoints)/sizeof(tDatums->at(0)->poseKeypoints[0]) << "\n";
                    // std::cout << "poseScores size: " << sizeof(tDatums->at(0)->poseScores)/sizeof(tDatums->at(0)->poseScores[0]) << "\n";
                    // std::cout << "poseCandidates size: " << tDatums->at(0)->poseCandidates.size() << "\n";
                    // std::cout << "poseKeypoints3D size: " << sizeof(tDatums->at(0)->poseKeypoints3D)/sizeof(tDatums->at(0)->poseKeypoints3D[0]) << "\n";
                    // std::cout << "poseNetOutput size: " << sizeof(tDatums->at(0)->poseNetOutput)/sizeof(tDatums->at(0)->poseNetOutput[0]) << "\n";
                    // std::cout << "netInputSizes size: " << tDatums->at(0)->netInputSizes.size() << "\n";
                    // std::cout << "netOutputSize: " << tDatums->at(0)->netOutputSize << "\n";

                /////////////////////////////////////////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////////
                /////////////////////////////////////////////////////////////////////////////////////////

                // std::cout << "---->wPoseExtractor work successful\n";
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WPoseExtractor);
}

#endif // OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP
