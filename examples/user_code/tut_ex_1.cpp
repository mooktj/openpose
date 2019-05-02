// ----------------------------- OpenPose C++ API Tutorial - Example 1 - Body from image -----------------------------
// It reads an image, process it, and displays it with the pose keypoints.

// Command-line user intraface
// #define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
// OpenPose dependencies
#include <openpose/headers.hpp>

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

        // Process and display image
        const auto imageToProcess = cv::imread(FLAGS_image_path);
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);
        if (datumProcessed != nullptr)
        {
            // printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
                display(datumProcessed);
        }
        else
            op::log("Image could not be processed.", op::Priority::High);

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