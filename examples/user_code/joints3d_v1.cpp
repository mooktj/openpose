/********** !!!!!! .......................................................................................
^^                          The aim of this script is to
                                extract heatmaps from OpenPose,
                                structure heatmaps data to make compatible to VNect input.
........................................................................................ !!!!!! **********/



// Command-line user intraface
// #define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/enumClasses.hpp>

// #include <openpose/printToFile.hpp>
#include <iostream>
#include <cstdio>

// Custom OpenPose flags
// Producer
DEFINE_string(image_path, "./examples/media/COCO_val2014_000000000192.jpg",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

DEFINE_string(moonbyul, "hello Moonbyul", "");


// void printToFile(const std::string& input)
void openLogFile()
{
    freopen("outputs/logs/Mamamoo.txt","a",stdout);
    std::cout << "\n\n\n\n";
    std::cout << "_____________________________________________________________________________________________________" << "\n";
    std::cout << "_____________________________________________________________________________________________________" << "\n";
    std::cout << "_____________________________________________________________________________________________________" << "\n";
    std::cout << "_____________________________________________________________________________________________________" << "\n";
}

void closeLogFile()
{
    fclose(stdout);
    std::cout << "Log file completed.Congratulations!" << "\n";
}

void printStar() {
    std::cout << "**\n";std::cout << "**\n";
}

bool displayHeatMaps(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, const int desiredChannel = 0)
{
    // std::cout << "----------------------------------------displayHeatMaps----------------------------------------" << "\n";
    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            auto& poseHeatMaps = datumsPtr->at(0)->poseHeatMaps;

            // std::cout << "****^^Mookie the Rookie Cookie Smoothie^^presents.--displayHeatMaps((->sizeof(poseHeatMaps): " << sizeof(poseHeatMaps) << "\n";

            const auto numberChannels = poseHeatMaps.getSize(0);
            const auto height = poseHeatMaps.getSize(1);
            const auto width = poseHeatMaps.getSize(2);
            const cv::Mat desiredChannelHeatMap(height, width, CV_32F, &poseHeatMaps.getPtr()[desiredChannel % numberChannels*height*width]);

            // std::cout << "-----------------------------------------set desiredChannelHeatMap----------------------------------------" << "\n";
            /*---------^^ Read input image of OpenPose body network ^^---------*/
            auto& inputNetData = datumsPtr->at(0)->inputNetData[0];
            const cv::Mat inputNetDataB(height, width, CV_32F, &inputNetData.getPtr()[0]);
            const cv::Mat inputNetDataG(height, width, CV_32F, &inputNetData.getPtr()[height*width]);
            const cv::Mat inputNetDataR(height, width, CV_32F, &inputNetData.getPtr()[2*height*width]);
            cv::Mat netInputImage;
            cv::merge(std::vector<cv::Mat>{inputNetDataB, inputNetDataG, inputNetDataR}, netInputImage);
            netInputImage = (netInputImage+0.5)*255;
            // std::cout << "-----------------------------------------set netInputImage----------------------------------------" << "\n";

            cv::Mat netInputImageUint8;
            cv::Mat desiredChannelHeatMapUint8;
            netInputImage.convertTo(netInputImageUint8, CV_8UC1);
            desiredChannelHeatMap.convertTo(desiredChannelHeatMapUint8, CV_8UC1);
            // std::cout << "-----------------------------------------converted netInputImage and desiredChannelHeatMap to Uint8----------------------------------------" << "\n";
            // std::cout << "**^^Mookie the Rookie Cookie Smoothie&&--presents--((->desiredChannelHeatMap.size(): " << desiredChannelHeatMap.size() << "\n";

            cv::Mat imageToRender;
            cv::applyColorMap(desiredChannelHeatMapUint8, desiredChannelHeatMapUint8, cv::COLORMAP_JET);
            cv::addWeighted(netInputImageUint8, 0.5, desiredChannelHeatMapUint8, 0.5, 0., imageToRender);
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " ^^Mookie the Rookie Cookie^^ DisplayHeatMap !XD", imageToRender);
        }
        else
        {
            op::log("Nullptr or empty datumsPtrs found.", op::Priority::High);
        }
        const auto key = (char)cv::waitKey(1);
        return (key==27);
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        return true;
    }
}






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
            // Display image
            cv::imshow(OPEN_POSE_NAME_AND_VERSION + " - Tutorial C++ API", datumsPtr->at(0)->cvOutputData);
            cv::waitKey(0);
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
            // op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);

            // // Alternative 2
            // op::log(datumsPtr->at(0).poseKeypoints, op::Priority::High);

            // // Alternative 3
            // std::cout << datumsPtr->at(0).poseKeypoints << std::endl;

            // // Alternative 4 - Accesing each element of the keypoints
            // op::log("\nKeypoints:", op::Priority::High);
            // const auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
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

            /*--------------- GET HEATMAPS IN ANY GIVEN OUTPUT FORM ---------------*/
            const auto& poseHMs = datumsPtr->at(0)->poseHeatMaps;
            std::cout << "*********Mook log:: poseHMs: " << poseHMs << "\n";
            /*--------------- END GET HEATMAPS IN ANY GIVEN OUTPUT FORM ---------------*/
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
            enableGoogleLogging };

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
        openLogFile();
        // printToFile("Welcome to Sanrio Land");
        // freopen("SavemetheLastDance.txt","w",stdout);
        op::log("----------------------------------------Starting OpenPose demo...", op::Priority::High);
        printStar();
        const auto opTimer = op::getTimerInit();

        // Required flags to enable heatmaps
        FLAGS_heatmaps_add_parts = true;
        FLAGS_heatmaps_add_bkg = true;
        FLAGS_heatmaps_add_PAFs = true;
        FLAGS_heatmaps_scale = 2;

        // Configuring OpenPose
        op::log("----------------------------------------Configuring OpenPose...", op::Priority::High);
        printStar();
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // std::cout << "--------opWrapper{op::ThreadManagerMode::Asynchronous} done,, about to configureWrapper(opWrapper)\n";
        configureWrapper(opWrapper);
        
        // Starting OpenPose
        op::log("----------------------------------------Starting thread(s)...", op::Priority::High);
        printStar();
        opWrapper.start();

        std::cout << "----------------------------------------opWrapper.started()" << "\n";
        printStar();

        // Process and display image
        const auto imageToProcess = cv::imread(FLAGS_image_path);
        std::cout << "----------------------------------------cv imread image path" << "\n";
        printStar();
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);

        std::cout << "----------------------------------------opWrapper emplaced and popped" << "\n";
        printStar();
        if (datumProcessed != nullptr)
        {
            /*--------------- GET HEATMAPS IN ANY GIVEN OUTPUT FORM ---------------*/
            // const auto& poseHMs = datumProcessed->at(0)->poseHeatMaps;
            // std::cout << "*********Mook log:: poseHMs: " << poseHMs << "\n";
            /*--------------- END GET HEATMAPS IN ANY GIVEN OUTPUT FORM ---------------*/

            // printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
            {
                // display(datumProcessed);
                const auto numberChannels = datumProcessed->at(0)->poseHeatMaps.getSize(0);
                // std::cout << "****^^Mookie the Rookie Cookie Smoothie presents^^" << "\n";
                // std::cout << "---------datumProcessed->at(0)->poseHeatMaps.getSize(0): " << datumProcessed->at(0)->poseHeatMaps.getSize(0) << "\n";
                // std::cout << "---------datumProcessed->at(0)->poseHeatMaps.getSize(1): " << datumProcessed->at(0)->poseHeatMaps.getSize(1) << "\n";
                // std::cout << "---------datumProcessed->at(0)->poseHeatMaps.getSize(2): " << datumProcessed->at(0)->poseHeatMaps.getSize(2) << "\n";
                
                for (auto desiredChannel = 0; desiredChannel < numberChannels; desiredChannel)
                {
                    if(displayHeatMaps(datumProcessed, desiredChannel))
                    {
                        std::cout << "----------------------------------------displayHeatMaps(datumProcessed, desiredChannel)" << "\n";
                        printStar();
                        break;
                    }    
                }                
            }
        }
        else
            op::log("Image could not be processed.", op::Priority::High);

        // const auto& poseBodyPartMappingBody25 = getPoseBodyPartMapping(op::PoseModel::BODY_25);
        // std::cout << "*********Mook log:: poseBodyPartMappingBody25 is " << typeid(poseBodyPartMappingBody25).name() << "\n";
        // const auto& poseBodyPartMappingCoco = getPoseBodyPartMapping(PoseModel::COCO_18);
        // const auto& poseBodyPartMappingMpi = getPoseBodyPartMapping(PoseModel::MPI_15);

        // Measuring total time
        op::printTime(opTimer, "OpenPose demo successfully finished. Total time: ", " seconds.", op::Priority::High);

        // std::cout << "*********Mook log:: FLAGS_write_json is " << FLAGS_write_json << "\n";
        // Return
        
        // std::cout<<"Hello Mamamoo Welcomes!<3\n";
        // fclose(stdout);
        closeLogFile();
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
