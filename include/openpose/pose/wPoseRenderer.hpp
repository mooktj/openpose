#ifndef OPENPOSE_POSE_W_POSE_RENDERER_HPP
#define OPENPOSE_POSE_W_POSE_RENDERER_HPP

#include <openpose/core/common.hpp>
#include <openpose/pose/poseRenderer.hpp>
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
    class WPoseRenderer : public Worker<TDatums>
    {
    public:
        explicit WPoseRenderer(const std::shared_ptr<PoseRenderer>& poseRendererSharedPtr);

        virtual ~WPoseRenderer();

        void initializationOnThread();

        void work(TDatums& tDatums);

        // float vNectFindMax(std::vector<float> in);

        // float vNectFindMin(std::vector<float> in);

    private:
        std::shared_ptr<PoseRenderer> spPoseRenderer;

        DELETE_COPY(WPoseRenderer);
    };
}

// Implementation
#include <openpose/utilities/pointerContainer.hpp>
#include <openpose/vnect/vNect.hpp>

namespace op
{
    template<typename TDatums>
    WPoseRenderer<TDatums>::WPoseRenderer(const std::shared_ptr<PoseRenderer>& poseRendererSharedPtr) :
        spPoseRenderer{poseRendererSharedPtr}
    {
    }

    template<typename TDatums>
    WPoseRenderer<TDatums>::~WPoseRenderer()
    {
    }

    template<typename TDatums>
    void WPoseRenderer<TDatums>::initializationOnThread()
    {
        try
        {
            spPoseRenderer->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    // float vNectFindMax(std::vector<float> in)
    // {
    //     float max = 0;
    //     for(unsigned int i = 0; i < in.size(); i++)
    //     {
    //         if(max < in.at(i)) max = in.at(i);
    //     }
    //     return max;
    // }

    // float vNectFindMin(std::vector<float> in)
    // {
    //     float min = 1000000;
    //     for(unsigned int i = 0; i < in.size(); i++)
    //     {
    //         if(min > in.at(i)) min = in.at(i);
    //     }
    //     return min;
    // }

    template<typename TDatums>
    void WPoseRenderer<TDatums>::work(TDatums& tDatums)
    {
        try
        {
            // bool renderOpenPose = false;

            if (checkNoNullNorEmpty(tDatums))
            {
                // std::cout << "...wPoseRenderer tDatums not nulll!!\n";
                // Debugging log
                dLog("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // Profiling speed
                const auto profilerKey = Profiler::timerInit(__LINE__, __FUNCTION__, __FILE__);
                // Render people pose
                for (auto& tDatumPtr : *tDatums)
                {
                    // cv::imshow("tDatumPtr outputDate before", tDatumPtr->cvOutputData);
                    // std::cout << "wPoseRenderer:: BEFORE: tDatumPtr->cvOutputData is empty? " << tDatumPtr->cvOutputData.empty() << "\n";
                    if(tDatumPtr->renderOpenPose) {
                        tDatumPtr->elementRendered = spPoseRenderer->renderPose(
                            tDatumPtr->outputData, tDatumPtr->poseKeypoints, (float)tDatumPtr->scaleInputToOutput,
                            (float)tDatumPtr->scaleNetToOutput);
                    }
                    // std::cout << "wPoseRenderer:: AFTER: tDatumPtr->cvOutputData is empty? " << tDatumPtr->cvOutputData.empty() << "\n";
                    // cv::imshow("tDatumPtr outputDate after", tDatumPtr->cvOutputData);
                    // std::cout << "elementRendered: " << tDatumPtr->elementRendered.first << ", " << tDatumPtr->elementRendered.second << "\n";
                    
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////                    
                    // **** VNect ADDED **** //
                    if(tDatumPtr->renderVNect) {
                        std::cout << "wPoseRenderer:: ~~Datum's renderVNect is ON\n";

                        auto& poseKeypoints = tDatumPtr->poseKeypoints;

                        std::vector<std::vector<float>> poses;
                        int numPoses = poseKeypoints.getSize(0);
                        int numEachPose = poseKeypoints.getSize(1)*poseKeypoints.getSize(2);

                        // GET X, Y LOCATIONS OF EACH POSE
                        std::cout << "numPoses: " << numPoses << "\n";
                        std::cout << "poseKeypoints.getSize(1): " << poseKeypoints.getSize(1) << ", getSize(2): " << poseKeypoints.getSize(2) << ", getSize(3): " << poseKeypoints.getSize(3) << "\n";
                        std::cout << "numEachPose: " << numEachPose << "\n";

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

                        cv::imshow("wPoseRenderer image", image);

                        for(int i = 0; i < posesBounds.size(); i++)
                        {
                            std::cout << "i: " << i << "\n";
                            std::cout << "left (x min): " << posesBounds.at(i).at(0) << "\n";
                            std::cout << "right (x max): " << posesBounds.at(i).at(1) << "\n";
                            std::cout << "bottom (y min): " << posesBounds.at(i).at(2) << "\n";
                            std::cout << "top (y max): " << posesBounds.at(i).at(3) << "\n";
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

                            cv::Mat noPadCropped = image(cv::Range(bottom,top), cv::Range(left,right));
                            cv::imshow(name, noPadCropped);


                            // vnect(croppedImg, vname);
                            std::string pathToWrite = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-1-5-19/";
                            vNectForward(croppedImg, vname, pathToWrite);
                        }

                        // vNectPostForward(tDatumPtr);

                        std::cout << "--------------------END VNECT--------------------\n";
                        // ----------- END VNECT ----------- //
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    }
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

                //     std::cout << "//////////////////PRINT DATUM INFO//////////////////\n";
                //     // std::cout << "size: " << tDatums->size() << "\n";
                //     // std::cout << "id: " << tDatums->at(0)->id << "\n";
                //     // std::cout << "cvInputData size: " << tDatums->at(0)->cvInputData.size() << "\n";
                //     // std::cout << "inputNetData size: " << tDatums->at(0)->inputNetData.size() << "\n";
                //     // std::cout << "outputData size: " << sizeof(tDatums->at(0)->outputData)/sizeof(tDatums->at(0)->outputData[0]) << "\n";
                //     // std::cout << "cvOutputData size: " << tDatums->at(0)->cvOutputData << "\n";
                //     // std::cout << "cvOutputData3D size: " << tDatums->at(0)->cvOutputData3D.size() << "\n";

                //     // std::cout << "poseKeypoints size: " << sizeof(tDatums->at(0)->poseKeypoints)/sizeof(tDatums->at(0)->poseKeypoints[0]) << "\n";
                //     // std::cout << "poseScores size: " << sizeof(tDatums->at(0)->poseScores)/sizeof(tDatums->at(0)->poseScores[0]) << "\n";
                //     // std::cout << "poseCandidates size: " << tDatums->at(0)->poseCandidates.size() << "\n";
                //     // std::cout << "poseKeypoints3D size: " << sizeof(tDatums->at(0)->poseKeypoints3D)/sizeof(tDatums->at(0)->poseKeypoints3D[0]) << "\n";
                //     // std::cout << "poseNetOutput size: " << sizeof(tDatums->at(0)->poseNetOutput)/sizeof(tDatums->at(0)->poseNetOutput[0]) << "\n";
                //     // std::cout << "netInputSizes size: " << tDatums->at(0)->netInputSizes.size() << "\n";
                //     // std::cout << "netOutputSize: " << tDatums->at(0)->netOutputSize << "\n";

                //     std::cout << "outputData getSize size: " << tDatums->at(0)->outputData.getSize().size() << "\n";
                //     std::cout << "outputData getSize 0: " << tDatums->at(0)->outputData.getSize().at(0) << "\n";
                //     std::cout << "outputData getSize 1: " << tDatums->at(0)->outputData.getSize().at(1) << "\n";
                //     std::cout << "outputData getSize 2: " << tDatums->at(0)->outputData.getSize().at(2) << "\n";
                //     std::cout << "outputData printSize: " << tDatums->at(0)->outputData.printSize() << "\n";

                //     std::cout << "sizeof(tDatums->at(0)->outputData): " << sizeof(tDatums->at(0)->outputData) << "\n";
                //     std::cout << "sizeof(tDatums->at(0)->outputData[0])" << sizeof(tDatums->at(0)->outputData[0]) << "\n";

                //     cv::Mat outputDataMat = tDatums->at(0)->outputData.getConstCvMat();
                //     std::cout << "outputDataMat size: " << outputDataMat.size() << "\n";
                //     std::cout << "outputDataMat rows: " << outputDataMat.rows << "\n";
                //     std::cout << "outputDataMat cols: " << outputDataMat.cols << "\n";
                //     std::cout << "outputDataMat channels: " << outputDataMat.channels() << "\n";
                //     std::cout << "outputDataMat depth: " << outputDataMat.depth() << "\n";

                //     std::vector<cv::Mat> bgrChannels;
                //     cv::split(outputDataMat, bgrChannels);
                //     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     /////////////WRITE OUTPUTDATAMAT.TXT///////////////////////////////////////////////////////////////////////////////////
                //     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_oic("outputs/wPoseRenderer/outputdatamat.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_oic = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_oic.rdbuf());
                //     //     std::cout << outputDataMat << "\n";
                //     // std::cout.rdbuf(coutbuf_oic);
                //     // std::cout << "written outputdatamat\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // ////////PRINT OUTPUTDATAMAT_C0.TXT////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_0("outputs/wPoseRenderer/outputdatamat_c0.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_0 = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_0.rdbuf());
                //     //     std::cout << bgrChannels.at(0) << "\n";
                //     // std::cout.rdbuf(coutbuf_0);
                //     // std::cout << "written outputdatamat_c0\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // ////////PRINT OUTPUTDATAMAT_C1.TXT////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_1("outputs/wPoseRenderer/outputdatamat_c1.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_1 = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_1.rdbuf());
                //     //     std::cout << bgrChannels.at(1) << "\n";
                //     // std::cout.rdbuf(coutbuf_1);
                //     // std::cout << "written outputdatamat_c1\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // ////////PRINT OUTPUTDATAMAT_C2.TXT////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_2("outputs/wPoseRenderer/outputdatamat_c2.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_2 = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_2.rdbuf());
                //     //     std::cout << bgrChannels.at(2) << "\n";
                //     // std::cout.rdbuf(coutbuf_2);
                //     // std::cout << "written outputdatamat_c2\n";
                //     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



                //     std::cout << "/////////----poseKeypoints/////////-------------------------\n";
                //     std::cout << "poseKeypoints getSize size: " << tDatums->at(0)->poseKeypoints.getSize().size() << "\n";
                //     std::cout << "poseKeypoints getSize 0: " << tDatums->at(0)->poseKeypoints.getSize().at(0) << "\n";
                //     std::cout << "poseKeypoints getSize 1: " << tDatums->at(0)->poseKeypoints.getSize().at(1) << "\n";
                //     std::cout << "poseKeypoints getSize 2: " << tDatums->at(0)->poseKeypoints.getSize().at(2) << "\n";
                //     std::cout << "poseKeypoints printSize: " << tDatums->at(0)->poseKeypoints.printSize() << "\n";

                //     std::cout << "sizeof(tDatums->at(0)->poseKeypoints): " << sizeof(tDatums->at(0)->poseKeypoints) << "\n";
                //     std::cout << "sizeof(tDatums->at(0)->poseKeypoints[0]): " << sizeof(tDatums->at(0)->poseKeypoints[0]) << "\n";

                //     cv::Mat poseKeypointsMat = tDatums->at(0)->poseKeypoints.getConstCvMat();
                //     std::cout << "poseKeypointsMat size: " << poseKeypointsMat.size() << "\n";
                //     std::cout << "poseKeypointsMat rows: " << poseKeypointsMat.rows << "\n";
                //     std::cout << "poseKeypointsMat cols: " << poseKeypointsMat.cols << "\n";
                //     std::cout << "poseKeypointsMat channels: " << poseKeypointsMat.channels() << "\n";
                //     std::cout << "poseKeypointsMat depth: " << poseKeypointsMat.depth() << "\n";

                //     // std::vector<cv::Mat> bgrChannelsPose;
                //     // cv::split(poseKeypointsMat, bgrChannelsPose);
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////WRITE poseKeypointsMAT.TXT///////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_oic_pose("outputs/wPoseRenderer/poseKeypointsmat.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_oic_pose = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_oic_pose.rdbuf());
                //     //     std::cout << poseKeypointsMat << "\n";
                //     // std::cout.rdbuf(coutbuf_oic_pose);
                //     // std::cout << "written poseKeypointsmat\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // ////////PRINT poseKeypointsMAT_C0.TXT////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_0_pose("outputs/wPoseRenderer/poseKeypointsmat_c0.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_0_pose = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_0_pose.rdbuf());
                //     //     std::cout << bgrChannelsPose.at(0) << "\n";
                //     // std::cout.rdbuf(coutbuf_0_pose);
                //     // std::cout << "written poseKeypointsmat_c0\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // ////////PRINT poseKeypointsMAT_C1.TXT////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_1_pose("outputs/wPoseRenderer/poseKeypointsmat_c1.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_1_pose = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_1_pose.rdbuf());
                //     //     std::cout << bgrChannelsPose.at(1) << "\n";
                //     // std::cout.rdbuf(coutbuf_1_pose);
                //     // std::cout << "written poseKeypointsmat_c1\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // ////////PRINT poseKeypointsMAT_C2.TXT////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // std::ofstream out_2_pose("outputs/wPoseRenderer/poseKeypointsmat_c2.txt"); //img[0][0][0]: -1278951244
                //     // std::streambuf *coutbuf_2_pose = std::cout.rdbuf();
                //     // std::cout.rdbuf(out_2_pose.rdbuf());
                //     //     std::cout << bgrChannelsPose.at(2) << "\n";
                //     // std::cout.rdbuf(coutbuf_2_pose);
                //     // std::cout << "written poseKeypointsmat_c2\n";
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                //     // /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////




                // /////////////////////////////////////////////////////////////////////////////////////////
                // /////////////////////////////////////////////////////////////////////////////////////////
                // /////////////////////////////////////////////////////////////////////////////////////////
                // /////////////////////////////////////////////////////////////////////////////////////////

                std::cout << "---->wPoseRenderer work successful\n";
            }
        }
        catch (const std::exception& e)
        {
            this->stop();
            tDatums = nullptr;
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    COMPILE_TEMPLATE_DATUM(WPoseRenderer);
}

#endif // OPENPOSE_POSE_W_POSE_RENDERER_HPP
