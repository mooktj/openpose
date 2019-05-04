#ifndef OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP
#define OPENPOSE_POSE_W_POSE_EXTRACTOR_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/poseExtractor.hpp>
#include <openpose/thread/worker.hpp>
#include <iostream>

#include <cstdio>
#include <fstream>
#include <cstring>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace op
{
    template<typename TDatums>
    class WPoseExtractor : public Worker<TDatums>
    {
    public:
        explicit WPoseExtractor(const std::shared_ptr<PoseExtractor>& poseExtractorSharedPtr, const bool vnectEnable);

        virtual ~WPoseExtractor();

        void initializationOnThread();

        void work(TDatums& tDatums);

    private:
        std::shared_ptr<PoseExtractor> spPoseExtractor;
        bool spVnectEnable;

        DELETE_COPY(WPoseExtractor);
    };
}





// Implementation
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/vnect/vNect.hpp>

namespace op
{
    template<typename TDatums>
    WPoseExtractor<TDatums>::WPoseExtractor(const std::shared_ptr<PoseExtractor>& poseExtractorSharedPtr, const bool vnectEnable) :
        spPoseExtractor{poseExtractorSharedPtr},
        spVnectEnable{vnectEnable}
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

                    // if(spVnectEnable)
                    // {
                    //     std::cout << "wPoseExtractor:: spVnectEnable!\n";
                    // }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                    
                    // // **** VNect ADDED **** //
                    //
                    //////////// NEED TO MIGRATE TO VNECT OWN CLASS ///////////////////////////////
                    //
                    // if(tDatumPtr->renderVNect) {
                    if(spVnectEnable)
                    {
                        // std::cout << "wPoseExtractor:: ~~Datum's renderVNect is ON\n";

                        auto& poseKeypoints = tDatumPtr->poseKeypoints;

                        std::vector<std::vector<float>> poses;
                        int numPoses = poseKeypoints.getSize(0);
                        int numEachPose = poseKeypoints.getSize(1)*poseKeypoints.getSize(2);

                        // GET X, Y LOCATIONS OF EACH POSE
                        // std::cout << "numPoses: " << numPoses << "\n";
                        // std::cout << "poseKeypoints.getSize(1): " << poseKeypoints.getSize(1) << ", getSize(2): " << poseKeypoints.getSize(2) << ", getSize(3): " << poseKeypoints.getSize(3) << "\n";
                        // std::cout << "numEachPose: " << numEachPose << "\n";

                        for(int pose = 0; pose < numPoses; pose++)
                        {
                            std::vector<float> posePoints_x;
                            std::vector<float> posePoints_y;

                            // std::cout << "    pose: " << pose << "\n";

                            for(int j = pose*numEachPose; j < (pose+1)*numEachPose; j += 3)
                            {
                                // std::cout << poseKeypoints.at(j) << ", " << poseKeypoints.at(j+1) << "\n";

                                posePoints_x.push_back(poseKeypoints.at(j));
                                posePoints_y.push_back(poseKeypoints.at(j+1));
                            }
                            poses.push_back(posePoints_x);
                            poses.push_back(posePoints_y);
                        }

                        // std::cout << "poses size: " << poses.size() << "\n";
                        // std::cout << "poses(0) size: " << poses.at(0).size() << "\n";

                        std::vector<std::vector<float>> posesBounds;
                        for(unsigned int i = 0; i < poses.size(); i += 2)
                        {
                            std::vector<float> bounds;
                            bounds.push_back(vNectFindMin(poses.at(i))); //x //need to deal with 0.00 ?
                            bounds.push_back(vNectFindMax(poses.at(i))); //x //need to deal with 0.00 ?
                            bounds.push_back(vNectFindMin(poses.at(i+1))); //y //need to deal with 0.00 ?
                            bounds.push_back(vNectFindMax(poses.at(i+1))); //y //need to deal with 0.00 ?
                            posesBounds.push_back(bounds);
                        }

                        cv::Mat image = tDatumPtr->cvInputData;

                        // cv::imshow("wPoseRenderer image", image);

                        for(int i = 0; i < (int)posesBounds.size(); i++)
                        {
                            // std::cout << "i: " << i << "\n";
                            // std::cout << "left (x min): " << posesBounds.at(i).at(0) << "\n";
                            // std::cout << "right (x max): " << posesBounds.at(i).at(1) << "\n";
                            // std::cout << "bottom (y min): " << posesBounds.at(i).at(2) << "\n";
                            // std::cout << "top (y max): " << posesBounds.at(i).at(3) << "\n";
                            float left = posesBounds.at(i).at(0);
                            float right = posesBounds.at(i).at(1);
                            float bottom = posesBounds.at(i).at(2);
                            float top = posesBounds.at(i).at(3);

                            float offset = 30;

                            float leftB = (left-offset < 0) ? 0 : left-offset;
                            float rightB = (right+offset > image.cols) ? image.cols : right+offset;
                            float bottomB = (bottom-offset < 0) ? 0 : bottom-offset;
                            float topB = (top+offset > image.rows) ? image.rows : top+offset;

                            cv::Mat croppedImg = image(cv::Range(bottomB,topB), cv::Range(leftB,rightB));
                            std::string vname = "media_2_croppedImg_" + std::to_string(i);
                            std::string name = "into Vnect " + vname;

                            // ORIGINAL CROPPED IMAGE WITHOUT NO PADDING
                            // cv::Mat noPadCropped = image(cv::Range(bottom,top), cv::Range(left,right));
                            // cv::imshow(name, noPadCropped);


                            // vnect(croppedImg, vname);
                            std::string pathToWrite = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-1-5-19/";
                            std::vector<std::vector<float>> joints_3d = vNectForward(croppedImg, vname, pathToWrite);
                            tDatumPtr->joints_3d_root_relative.push_back(joints_3d);
                        }

                        // vNectPostForward(tDatumPtr);

                        // std::cout << "--------------------END VNECT--------------------\n";
                        // ----------- END VNECT ----------- //
                    }
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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
