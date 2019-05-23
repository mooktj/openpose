// ----------------------------- OpenPose C++ API Tutorial - Example 1 - Body from image -----------------------------
// It reads an image, process it, and displays it with the pose keypoints.

// Command-line user intraface
// #define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>
#include <string>


#include <ncurses.h>
#include <python2.7/Python.h>

#include <openpose/OpVNectPostFunctions.cpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Custom OpenPose flags
// Producer
// DEFINE_string(image_path, "examples/media/COCO_val2014_000000000192.jpg",
//     "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_string(image_path, "./examples/media/multi-person/media_2.png",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

// This worker will just read and return all the jpg files in a directory
void display(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // User's displaying/saving/other processing here
            // datum.cvOutputData: rendered frame with pose or heatmaps
            // datum.poseKeypoints: Array<float> with the estimated pose
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // auto& joints3d = datumsPtr->at(0)->joints_3d_root_relative;
            // std::cout << "joints3d size: " << joints3d.size() << "\n";
            // for(int i = 0; i < joints3d.size(); i++)
            // {
            //     std::cout << "i: " << i << "\n";
            //     for(int j = 0; j < joints3d.at(i).size(); j++)
            //     {
            //         std::cout << joints3d.at(i).at(j).at(0) << ", " << joints3d.at(i).at(j).at(1) << ", " << joints3d.at(i).at(j).at(2) << "\n";
            //     }
            // }

            cv::Mat cvOutput = datumsPtr->at(0)->cvOutputData;
            auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            // std::cout << "poseKeypoints size: " << poseKeypoints.getSize() << "\n";
            // std::cout << "poseKeypoints getSize(0): " << poseKeypoints.getSize(0) << "\n";
            // std::cout << "poseKeypoints getSize(1): " << poseKeypoints.getSize(1) << "\n";
            // std::cout << "poseKeypoints getSize(2): " << poseKeypoints.getSize(2) << "\n";

            int fontFace = CV_FONT_HERSHEY_SIMPLEX;

            for(auto person = 0; person < poseKeypoints.getSize(0); person++)
            {
                // std::cout << "person: " << person << " ----> ";

                int bodyPart = 0;
                auto bodyPart_index = (person * poseKeypoints.getSize(1) * poseKeypoints.getSize(2)) + (poseKeypoints.getSize(2) * bodyPart);
                // std::cout << "bodyPart_index: " << bodyPart_index << ", ";
                // std::cout << "" << poseKeypoints.at(bodyPart_index) << "\n";
                cv::Point p1 = cv::Point((int) poseKeypoints.at(bodyPart_index) , (int) poseKeypoints.at(bodyPart_index + 1));
                cv::Point p2 = cv::Point(p1.x+30, p1.y-30);
                cv::line(cvOutput, p1, p2, cv::Scalar(255,0,255), 1, 8, 0);
                std::string text = "pose_" + std::to_string(person);
                cv::Point textOrg = cv::Point(p1.x+35, p1.y-35);
                cv::putText(cvOutput, text, textOrg, fontFace, 0.5, cv::Scalar(255,0,255), 1, 8);
            }



            // Display image
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", cvOutput);
            // cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", datumsPtr->at(0)->cvOutputData);
            // cv::waitKey(0);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void printKeypoints(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try
    {
        // Example: How to use the pose keypoints
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            // Alternative 1
            op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);

            // // Alternative 2
            // op::log(datumsPtr->at(0).poseKeypoints, op::Priority::High);

            // // Alternative 3
            // std::cout << datumsPtr->at(0).poseKeypoints << std::endl;

            // // Alternative 4 - Accesing each element of the keypoints
            // op::log("\nKeypoints:", op::Priority::High);
            // const auto& poseKeypoints = datumsPtr->at(0).poseKeypoints;
            // op::log("Person pose keypoints:", op::Priority::High);
            // for (auto person = 0 ; person < poseKeypoints.getSize(0) ; person++)
            // {
            //     op::log("Person " + std::to_string(person) + " (x, y, score):", op::Priority::High);
            //     for (auto bodyPart = 0 ; bodyPart < poseKeypoints.getSize(1) ; bodyPart++)
            //     {
            //         std::string valueToPrint;
            //         for (auto xyscore = 0 ; xyscore < poseKeypoints.getSize(2) ; xyscore++)
            //             valueToPrint += std::to_string(   poseKeypoints[{person, bodyPart, xyscore}]   ) + " ";
            //         op::log(valueToPrint, op::Priority::High);
            //     }
            // }
            // op::log(" ", op::Priority::High);
        }
        else
            op::log("Nullptr or empty datumsPtr found.", op::Priority::High);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void configureWrapper(op::Wrapper& opWrapper)
{
    try
    {
        op::check(0 <= FLAGS_logging_level && FLAGS_logging_level <= 255, "Wrong logging_level value.",
                    __LINE__, __FUNCTION__, __FILE__);
        op::ConfigureLog::setPriorityThreshold((op::Priority)FLAGS_logging_level);
        op::Profiler::setDefaultX(FLAGS_profile_speed);

        FLAGS_model_pose = "MPI";
        // FLAGS_model_pose = "BODY_25";
        // FLAGS_part_candidates = true;

        // Apply GFlags to program variables
        const auto poseMode = op::flagsToPoseMode(FLAGS_body);
        const auto netInputSize = op::flagsToPoint(FLAGS_net_resolution, "-1x368");
        const auto outputSize = op::flagsToPoint(FLAGS_output_resolution, "-1x-1");
        const auto keypointScaleMode = op::flagsToScaleMode(FLAGS_keypoint_scale);
        const auto gpuNumber = FLAGS_num_gpu;
        const auto gpuNumberStart = FLAGS_num_gpu_start;
        const auto scalesNumber = FLAGS_scale_number;
        const auto scaleGap = (float)FLAGS_scale_gap;
            const auto multipleView = (FLAGS_3d || FLAGS_3d_views > 1);
        const auto renderMode = op::flagsToRenderMode(FLAGS_render_pose, multipleView);
        const auto poseModel = op::flagsToPoseModel(FLAGS_model_pose);
        const auto blendOriginalFrame = !FLAGS_disable_blending; // true: does blend on original image
        const auto alphaKeypoint = (float)FLAGS_alpha_pose;
        const auto alphaHeatMap = (float)FLAGS_alpha_heatmap;
        const auto defaultPartToRender = FLAGS_part_to_show;
        const auto modelFolder = FLAGS_model_folder;
        const auto heatMapTypes = op::flagsToHeatMaps(FLAGS_heatmaps_add_parts, FLAGS_heatmaps_add_bkg, FLAGS_heatmaps_add_PAFs);
        const auto heatMapScaleMode = op::flagsToHeatMapScaleMode(FLAGS_heatmaps_scale);
        const auto addPartCandidates = FLAGS_part_candidates;
        const auto renderThreshold = (float)FLAGS_render_threshold;
        const auto numberPeopleMax = FLAGS_number_people_max;
        const auto maximizePositives = FLAGS_maximize_positives;
        const auto fpsMax = FLAGS_fps_max;
        const auto protoTxtPath = FLAGS_prototxt_path;
        const auto caffeModelPath = FLAGS_caffemodel_path;
        const auto upsamplingRatio = (float)FLAGS_upsampling_ratio;
        const auto enableGoogleLogging = true;

        // const auto vnectEnable = true;
        // const auto vModelFolder = "";
        // const auto vProtoTxtFile = "";
        // const auto vTrainedModelFile = "";

        // const op::WrapperStructVnect wrapperStructVnect{
        //     vnectEnable,
        //     vModelFolder,
        //     vProtoTxtFile,
        //     vTrainedModelFile
        // };

        // opWrapper.configure(wrapperStructVnect);


        // std::cout << "keypoints_ordering:: caffeModelPath = " << caffeModelPath << "\n";
        // Set up WrapperStructPose
        const op::WrapperStructPose wrapperStructPose{
            poseMode, 
            netInputSize, 
            outputSize,
            keypointScaleMode, 
            gpuNumber, 
            gpuNumberStart, 
            scalesNumber,
            (float)scaleGap, 
            renderMode, 
            poseModel,
            blendOriginalFrame, 
            (float)alphaKeypoint, 
            (float)alphaHeatMap,
            defaultPartToRender, 
            modelFolder, 
            heatMapTypes,
            heatMapScaleMode, 
            addPartCandidates, 
            (float)renderThreshold,
            numberPeopleMax, 
            maximizePositives, 
            fpsMax,
            protoTxtPath, 
            caffeModelPath, 
            (float)upsamplingRatio,
            enableGoogleLogging,
            FLAGS_vnect_set };

        opWrapper.configure(wrapperStructPose);

        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();


    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

int tutorialApiCpp()
{
    try
    {

        // std::cout << "OpenCV version: " << CV_VERSION << std::endl;
        // op::log("Starting OpenPose demo...", op::Priority::High);
        const auto opTimer = op::getTimerInit();

        // Configuring OpenPose
        // op::log("Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};

        configureWrapper(opWrapper);
        // Set to single-thread (for sequential processing and/or debugging and/or reducing latency)
        if (FLAGS_disable_multi_thread)
            opWrapper.disableMultiThreading();

        // Starting OpenPose
        // op::log("Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        std::vector<std::string> images;
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_3884.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_0.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_1018.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_1025.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_1037.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_1114.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_408.jpg");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_489.jpg");
        images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_1.png");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_2.png");
        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_3.png");


        // images.push_back("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_489.jpg");
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////// WRITING VIDEO /////////////////////////////////////////////////////////////////////////
        // cv::VideoCapture cap("/home/mooktj/Desktop/myworkspace/Gogobebe_dance_cover.mp4");
        // if(!cap.isOpened())
        // {
        //     std::cout << "ERROR OPENING VIDEO STREAM/FILE!\n";
        //     exit(1);
        // }

        // // const std::string NAME = "Gogobebe_dance_cover.avi";
        // // // int ex = static_cast<int>(cap.get(CV_CAP_PROP_FOURCC)); 

        // // int frame_width = (int) cap.get(CV_CAP_PROP_FRAME_WIDTH);
        // // int frame_height = (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT);
        // // cv::Size frame_size(frame_width, frame_height);
        // // int frames_per_sec = 10;

        // // // cv::VideoWriter outputVideo(NAME, cv::VideoWriter::CV_FOURCC('P','I','M','1'), cap.get(CV_CAP_PROP_FPS), frame_size, true);

        // // cv::Size S = cv::Size((int) cap.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
        // //           (int) cap.get(CV_CAP_PROP_FRAME_HEIGHT));

        // // cv::VideoWriter outputVideo;
        // // // outputVideo.set(CV_CAP_PROP_FOURCC, CV_FOURCC('P','I','M','1'));
        // // outputVideo.open(NAME, CV_FOURCC('M','J','P','G'), cap.get(CV_CAP_PROP_FPS), S, true);
        // // if(!outputVideo.isOpened())
        // // {
        // //     std::cout << "ERROR CREATING/OPENING OUTPUTVIDEO TO WRITE!\n";
        // //     exit(1);
        // // }

        // int i = 0;
        // while(true)
        // {
        //     std::cout << "processing\n";
        //     cv::Mat frame;
        //     cap >> frame;
        //     if(frame.empty())
        //     {
        //         std::cout << "Video has finished.\n";
        //         break;
        //     }

        //     if((char)cv::waitKey(25) == 27)
        //     {
        //         std::cout << "Exit Video.\n";
        //         break;
        //     }

        //     // const auto imageToProcess = cv::imread(frame);
        //     auto datumProcessed = opWrapper.emplaceAndPop(frame);

        //     if (datumProcessed != nullptr)
        //     {
        //         // printKeypoints(datumProcessed);
        //         write3dJointsVNect(datumProcessed, i++);
        //         // matplotlibVNect();

        //         // if (!FLAGS_no_display)
        //             // display(datumProcessed);
        //             // outputVideo << datumProcessed->at(0)->cvOutputData;
        //             // cv::waitKey(300);
        //     }
        //     else
        //         op::log("Image could not be processed.", op::Priority::High);    

        // }

        // cap.release();
        // outputVideo.release();
        // cv::destroyAllWindows();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////INITIALISE VIDEO WRITER////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////
        // std::string pathToFrame = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_" + std::to_string(0) + ".jpg";
        // std::string pathToFrame = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/MMM_starry_night/frame_" + std::to_string(2362) + ".jpg";
        std::string pathToFrame = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/BTS_BwL/frame_" + std::to_string(0) + ".jpg";
        // std::string pathToFrame = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/mmm_" + std::to_string(12) + ".jpg";
        // std::string pathToFrame = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_1.png";
        const auto firstFrame = cv::imread(pathToFrame);

        cv::Size S = cv::Size((int) firstFrame.cols, (int) firstFrame.rows);
        std::string NAME = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-23-5-19/BwL_1800_2000/";
        NAME += "BwL_1800_2000.avi";

        cv::VideoWriter outputVideo;
        outputVideo.open(NAME, CV_FOURCC('M','J','P','G'), 1.0, S, true);

        if(!outputVideo.isOpened())
        {
            std::cout << "ERROR CREATING/OPENING OUTPUTVIDEO TO WRITE!\n";
            exit(1);
        }
        ///////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////


        std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>> datumProcessed;
        bool FIRST_TIME_IN = true;

        for(int i = 1800; i < 2000; i++)
        {
            std::cout << "----------------------------image i = " << i << "----------------------------\n";
            // Process and display image
            // std::string pathToImage = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/Gogobebe/frame_" + std::to_string(i) + ".jpg";
            // std::string pathToImage = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/MMM_starry_night/frame_" + std::to_string(i) + ".jpg";
            std::string pathToImage = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/BTS_BwL/frame_" + std::to_string(i) + ".jpg";
            
            // std::string pathToImage = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/mmm_" + std::to_string(i) + ".jpg";
            // std::string pathToImage = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_3.png";
            // const auto imageToProcess = cv::imread(images.at(i));
            const auto imageToProcess = cv::imread(pathToImage);
            if(FIRST_TIME_IN)
            {
                datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
                FIRST_TIME_IN = false;
                // std::cout << "----------------FIRST_TIME_IN----------------\n";
                // std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>

                // std::cout << "FIRST_TIME_IN-----i = " << i << "\n";
                // std::cout << "display Datum info.:\n";
                // std::cout << "  datumProcessed->snowmen\n";
                // for(auto s = 0; s < datumProcessed->at(0)->snowmen.size(); s++)
                // {
                //     std::cout << "s: " << s << "\n";
                //     // for(auto ss = 0; ss < datumProcessed->at(0)->snowmen.at(s).size(); ss++)
                //     // {
                //         std::cout << "\t 0. " << datumProcessed->at(0)->snowmen.at(s).at(0) << "\n";
                //         std::cout << "\t 1. " << datumProcessed->at(0)->snowmen.at(s).at(1) << "\n";
                //         std::cout << "\t 2. " << datumProcessed->at(0)->snowmen.at(s).at(2) << "\n";
                //         std::cout << "\t 3. " << datumProcessed->at(0)->snowmen.at(s).at(3) << "\n";
                //         std::cout << "\t 4. " << datumProcessed->at(0)->snowmen.at(s).at(4) << "\n";
                //         std::cout << "\t 5. " << datumProcessed->at(0)->snowmen.at(s).at(5) << "\n";
                //         std::cout << "\t 6. " << datumProcessed->at(0)->snowmen.at(s).at(6) << "\n";
                //         std::cout << "\t 7. " << datumProcessed->at(0)->snowmen.at(s).at(7) << "\n";
                //         std::cout << "\t 8. " << datumProcessed->at(0)->snowmen.at(s).at(8) << "\n";
                //     // }
                // }
                // std::cout << "  ---->datumProcessed->prevSnowmen\n";
                // for(auto s = 0; s < datumProcessed->at(0)->prevSnowmen.size(); s++)
                // {
                //     std::cout << "s: " << s << "\n";
                //     // for(auto ss = 0; ss < datumProcessed->at(0)->prevSnowmen.at(s).size(); ss++)
                //     // {
                //         std::cout << "\t 0. " << datumProcessed->at(0)->prevSnowmen.at(s).at(0) << "\n";
                //         std::cout << "\t 1. " << datumProcessed->at(0)->prevSnowmen.at(s).at(1) << "\n";
                //         std::cout << "\t 2. " << datumProcessed->at(0)->prevSnowmen.at(s).at(2) << "\n";
                //         std::cout << "\t 3. " << datumProcessed->at(0)->prevSnowmen.at(s).at(3) << "\n";
                //         std::cout << "\t 4. " << datumProcessed->at(0)->prevSnowmen.at(s).at(4) << "\n";
                //         std::cout << "\t 5. " << datumProcessed->at(0)->prevSnowmen.at(s).at(5) << "\n";
                //         std::cout << "\t 6. " << datumProcessed->at(0)->prevSnowmen.at(s).at(6) << "\n";
                //         std::cout << "\t 7. " << datumProcessed->at(0)->prevSnowmen.at(s).at(7) << "\n";
                //         std::cout << "\t 8. " << datumProcessed->at(0)->prevSnowmen.at(s).at(8) << "\n";
                //     // }
                // }

                // std::cout << "  ---->datumProcessed->orientation\n";
                // for(auto i = 0; i < datumProcessed->at(0)->orientation.size(); i+=2)
                // {
                //     std::cout << "i: " << i << ", x: " << datumProcessed->at(0)->orientation.at(i) << ", y: " << datumProcessed->at(0)->orientation.at(i+1) << "\n";
                // }

                // std::cout << "  ---->datumProcessed->prevOrientation\n";
                // for(auto i = 0; i < datumProcessed->at(0)->prevOrientation.size(); i+=2)
                // {
                //     std::cout << "i: " << i << ", x: " << datumProcessed->at(0)->prevOrientation.at(i) << ", y: " << datumProcessed->at(0)->prevOrientation.at(i+1) << "\n";
                // }

            }
            else
            {
                // datumProcessed->at(0)->cvInputData.release();
                // datumProcessed->at(0)->cvInputData = imageToProcess;
                datumProcessed = opWrapper.reEmplaceAndPop(datumProcessed, imageToProcess);   

                // std::cout << "          -----i = " << i << "\n";
                // std::cout << "display Datum info.:\n";
                // std::cout << "  datumProcessed->snowmen\n";
                // for(auto s = 0; s < datumProcessed->at(0)->snowmen.size(); s++)
                // {
                //     std::cout << "s: " << s << "\n";
                //     // for(auto ss = 0; ss < datumProcessed->at(0)->snowmen.at(s).size(); ss++)
                //     // {
                //         std::cout << "\t 0. " << datumProcessed->at(0)->snowmen.at(s).at(0) << "\n";
                //         std::cout << "\t 1. " << datumProcessed->at(0)->snowmen.at(s).at(1) << "\n";
                //         std::cout << "\t 2. " << datumProcessed->at(0)->snowmen.at(s).at(2) << "\n";
                //         std::cout << "\t 3. " << datumProcessed->at(0)->snowmen.at(s).at(3) << "\n";
                //         std::cout << "\t 4. " << datumProcessed->at(0)->snowmen.at(s).at(4) << "\n";
                //         std::cout << "\t 5. " << datumProcessed->at(0)->snowmen.at(s).at(5) << "\n";
                //         std::cout << "\t 6. " << datumProcessed->at(0)->snowmen.at(s).at(6) << "\n";
                //         std::cout << "\t 7. " << datumProcessed->at(0)->snowmen.at(s).at(7) << "\n";
                //         std::cout << "\t 8. " << datumProcessed->at(0)->snowmen.at(s).at(8) << "\n";
                //     // }
                // }

                // std::cout << "  ---->datumProcessed->prevSnowmen\n";
                // for(auto s = 0; s < datumProcessed->at(0)->prevSnowmen.size(); s++)
                // {
                //     std::cout << "s: " << s << "\n";
                //     // for(auto ss = 0; ss < datumProcessed->at(0)->prevSnowmen.at(s).size(); ss++)
                //     // {
                //         std::cout << "\t 0. " << datumProcessed->at(0)->prevSnowmen.at(s).at(0) << "\n";
                //         std::cout << "\t 1. " << datumProcessed->at(0)->prevSnowmen.at(s).at(1) << "\n";
                //         std::cout << "\t 2. " << datumProcessed->at(0)->prevSnowmen.at(s).at(2) << "\n";
                //         std::cout << "\t 3. " << datumProcessed->at(0)->prevSnowmen.at(s).at(3) << "\n";
                //         std::cout << "\t 4. " << datumProcessed->at(0)->prevSnowmen.at(s).at(4) << "\n";
                //         std::cout << "\t 5. " << datumProcessed->at(0)->prevSnowmen.at(s).at(5) << "\n";
                //         std::cout << "\t 6. " << datumProcessed->at(0)->prevSnowmen.at(s).at(6) << "\n";
                //         std::cout << "\t 7. " << datumProcessed->at(0)->prevSnowmen.at(s).at(7) << "\n";
                //         std::cout << "\t 8. " << datumProcessed->at(0)->prevSnowmen.at(s).at(8) << "\n";
                //     // }
                // }

                // std::cout << "  ---->datumProcessed->orientation\n";
                // for(auto i = 0; i < datumProcessed->at(0)->orientation.size(); i+=2)
                // {
                //     std::cout << "i: " << i << ", x: " << datumProcessed->at(0)->orientation.at(i) << ", y: " << datumProcessed->at(0)->orientation.at(i+1) << "\n";
                // }

                // std::cout << "  ---->datumProcessed->prevOrientation\n";
                // for(auto i = 0; i < datumProcessed->at(0)->prevOrientation.size(); i+=2)
                // {
                //     std::cout << "i: " << i << ", x: " << datumProcessed->at(0)->prevOrientation.at(i) << ", y: " << datumProcessed->at(0)->prevOrientation.at(i+1) << "\n";
                // }
            }             
            


            if (datumProcessed != nullptr)
            {
                // std::cout << "datumProcessed i = " << i << "\n";
                // printKeypoints(datumProcessed);
                // write3dJointsVNect(datumProcessed, pathToWrite, "frame_0");
                // matplotlibVNect();

                ///////////////////////////////////////////////////////////////////////////////////////////////
                //////////////////////WRITE VIDEO//////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////
                // std::cout << "==============> datumProcessed is not null!!\n";
                cv::Mat currPose = datumProcessed->at(0)->cvOutputData.clone();
                cv::Mat currPose_output = datumProcessed->at(0)->cvOutputData.clone();
                float opacity = 0.5;

                std::vector<std::vector<float>> snowmen = datumProcessed->at(0)->snowmen;
                std::vector<std::vector<float>> prevSnowmen = datumProcessed->at(0)->prevSnowmen;


                int fontFace = CV_FONT_HERSHEY_SIMPLEX;
                // std::cout << "snowmen size: " << snowmen.size() << "\n";
                for(auto c = 0; c < snowmen.size(); c++)
                {
                    std::cout << "c: " << c << "\n";
                    int cen_x = snowmen.at(c).at(0);
                    // std::cout << "-------check 2 " << cen_x << "\n";
                    float top_of_head = snowmen.at(c).at(1);
                    float bot_of_head = snowmen.at(c).at(2);
                    float top_of_mid = snowmen.at(c).at(3);
                    float bot_of_mid = snowmen.at(c).at(4);
                    float top_of_leg = snowmen.at(c).at(5);
                    float bot_of_leg = snowmen.at(c).at(6);
                    float x_left_bot_a = snowmen.at(c).at(7);
                    float y_left_bot_a = snowmen.at(c).at(8);

                    float rad_top =  bot_of_head - (top_of_head + bot_of_head) / 2;
                    float rad_mid =  bot_of_mid - (top_of_mid + bot_of_mid) / 2;
                    float rad_bot = bot_of_leg - (top_of_leg + bot_of_leg) / 2;

                    // DRAW SNOWMAN ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    cv::circle(currPose, cv::Point(cen_x, (int) (top_of_head + bot_of_head)/2), rad_top, cv::Scalar(0,0,255), -1, 8);
                    cv::circle(currPose, cv::Point(cen_x, (int) (top_of_mid + bot_of_mid)/2), rad_mid, cv::Scalar(0,0,255), -1, 8);
                    cv::circle(currPose, cv::Point(cen_x, (int) (top_of_leg + bot_of_leg)/2), rad_bot, cv::Scalar(0,0,255), -1, 8);

                    // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
                    // cv::circle(currPose, cv::Point(cen_x, (int)top_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
                    // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
                    // cv::circle(currPose, cv::Point(cen_x, (int)top_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
                    // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_head), 5, cv::Scalar(0,255,255), -1, 8);
                    // cv::circle(currPose, cv::Point(cen_x, (int)top_of_head), 5, cv::Scalar(0,255,255), -1, 8);

                    // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
                    //////////////////////////////////////////////////////////////////////////////////////
                    std::cout << "Height: " << snowmen.at(c).at(28) << "\n";

                    std::string tex = "depth " + std::to_string(c) + " : " + std::to_string(snowmen.at(c).at(27) * (1));
                    cv::Point textOrg = cv::Point(cen_x + (c*5), top_of_head - (11 * c));
                    cv::putText(currPose_output, tex, textOrg, fontFace, 0.4, cv::Scalar(0,0,0), 1, 8);  


                }

                cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
                // std::cout << "--> PRINT DEPTH <--\n";
                // for(auto c = 0; c < snowmen.size(); c++)
                // {
                //     std::cout << "c: " << c << "\n";
                //     std::cout << "  --------> depth: " << snowmen.at(c).at(27) * (1000000) << "\n";
                // }
                ///// WRITE FRAME NUMBER /////
                // std::string text = "frame " + std::to_string(i);
                std::string text = "frame_" + std::to_string(i);
                cv::Point textOrg = cv::Point(S.width-100, S.height-20);
                cv::putText(currPose_output, text, textOrg, fontFace, 0.5, cv::Scalar(255,0,255), 1, 8);

                // // std::cout << "==============> frame written!!\n";
                // // outputVideo << datumProcessed->at(0)->cvOutputData;
                ///////////////////////////////////////////////////////////////////////////////////////////////
                ///////////////////////////////////////////////////////////////////////////////////////////////

                // if (!FLAGS_no_display)
                //     display(datumProcessed);

                //////////////////////////////
                cv::Point f1(0, (int) datumProcessed->at(0)->floorLevelPt.y);
                cv::Point f2(S.width, (int) datumProcessed->at(0)->floorLevelPt.y);
                cv::line(currPose_output, f1, f2, cv::Scalar(0,255,0), 5, 8, 0);


                    outputVideo << currPose_output;
                    // cv::imshow("currPose_output", currPose_output);
                    // std::string pathToSave = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-21-5-19/GGBB/frame_" + std::to_string(i) + ".jpg";

                    // cv::imwrite(pathToSave, currPose_output);
                // std::cout << "----------> finished 1 datum\n";

                std::string pathToWrite = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-23-5-19/BwL_1800_2000/";
                std::string depthFileName = "frame_" + std::to_string(i) + "_depths";
                print3Ddepths(datumProcessed, pathToWrite, depthFileName);
                std::string pose3dFileName = "frame_" + std::to_string(i) + "_poses";
                write3dJointsVNect(datumProcessed, pathToWrite, pose3dFileName);
                    // cv::waitKey(1000);
            }
            else
                op::log("Image could not be processed.", op::Priority::High);     

            // std::cout << "--------------------------------------------------------\n--------------------------------------------------------\n";       
        }

        outputVideo.release();
        // cv::waitKey(0);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // Return
        return 0;
    }
    catch (const std::exception& e)
    {
        return -1;
    }
}

int main(int argc, char *argv[])
{
    // Parsing command line flags
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Running tutorialApiCpp
    return tutorialApiCpp();
}
