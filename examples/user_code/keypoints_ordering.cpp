/********** !!!!!! .......................................................................................
^^                          The aim of this script is to
                                experiment on the data from OpenPose,
                                structure output data to make compatible to VNect x,y,z heatmaps.
........................................................................................ !!!!!! **********/



// Command-line user intraface
#define PY_SSIZE_T_CLEAN
// #define OPENPOSE_FLAGS_DISABLE_POSE
#include <openpose/flags.hpp>
#include <openpose/headers.hpp>

#include <openpose/pose/poseParameters.hpp>
#include <openpose/pose/enumClasses.hpp>

// #include <openpose/printToFile.hpp>
#include <iostream>
#include <cstdio>

#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <string>
#include <fstream>
#include <cstring>

#include <math.h>

#include <stdio.h>
// #include <conio.h>
#include <ncurses.h>
#include <python2.7/Python.h>

// Custom OpenPose flags
// Producer
// DEFINE_string(image_path, "./examples/media/COCO_val2014_000000000192.jpg",
//     "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
DEFINE_string(image_path, "./examples/media/multi-person/media_2.png",
    "Process an image. Read all standard formats (jpg, png, bmp, etc.).");
// DEFINE_string(image_path, "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/vnect_media/frame899.jpg",
//     "Process an image. Read all standard formats (jpg, png, bmp, etc.).");

// Display
DEFINE_bool(no_display,                 false,
    "Enable to disable the visual display.");

DEFINE_string(moonbyul, "hello Moonbyul", "");

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
            cv::imwrite("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/drawPose2D/media_2.jpg", datumsPtr->at(0)->cvOutputData);
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

float findMax(std::vector<float> in)
{
    float max = 0;
    for(int i = 0; i < in.size(); i++)
    {
        if(max < in.at(i)) max = in.at(i);
    }
    return max;
}

float findMin(std::vector<float> in)
{
    float min = 1000000;
    for(int i = 0; i < in.size(); i++)
    {
        if(min > in.at(i)) min = in.at(i);
    }
    return min;
}

void vnect(cv::Mat croppedImg, std::string croppedImgName)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);

    const std::string modelFolder = "./models/";
    const std::string protoTxtfile = modelFolder + "vnect/vnect_net.prototxt";
    const std::string trainedModelFile = modelFolder + "vnect/vnect_model.caffemodel";

    float scales[3] = {1.0, 0.8, 0.6};
    int scales_size = sizeof(scales)/sizeof(scales[0]);

    std::cout << protoTxtfile << "\n";
    std::cout << trainedModelFile << "\n";
    std::shared_ptr<caffe::Net<float>> testnet;
    testnet.reset(new caffe::Net<float>(protoTxtfile, caffe::TEST));
    testnet->CopyTrainedLayersFrom(trainedModelFile);
    
    caffe::Blob<float>* input_layer = testnet->input_blobs()[0];

    int num_channels = input_layer->channels();
    // std::cout << "num_channels = " << input_layer->channels() << "\n";

    cv::Size input_geometry;
    input_geometry = cv::Size(input_layer->width(), input_layer->height());

    cv::Mat img = croppedImg;

    input_layer->Reshape(scales_size, num_channels, input_geometry.height, input_geometry.width);
    testnet->Reshape();

    int width = input_layer->width();
    int height = input_layer->height();

    cv::Mat sample_img;
    cv::Mat img_square;
    cv::Mat sample;

    sample_img = img.clone();
    img_square = img.clone();

    /* READ SQUARE IMAGE */
    float scale_square = (height * 1.0) / (img.rows * 1.0);
    cv::resize(img_square, sample_img, cv::Size(0,0), scale_square, scale_square, cv::INTER_LANCZOS4);
    std::cout << "img_square.size(): " << img_square.size() << "\n";
    std::cout << "sample_img.size(): " << sample_img.size() << "\n";

    if(sample_img.cols < width)
    {
        int offset = sample_img.cols % 2;
        sample = sample_img;

        int length_dst = width;
        int half_dst = std::floor(length_dst/2);

        int length_src = sample_img.cols;
        int half_src = std::floor(length_src/2);
        int remainder_src = length_src % 2;

        // int height_src = std::floor(sample_img.rows/2);

        int half_row =  std::floor(sample_img.rows/2);
        int remainder_row = sample_img.rows % 2;
        int top = half_dst - half_row;
        int bottom = top - remainder_row;

        int start = (half_dst - half_src);
        int end = (half_dst + half_src) + remainder_src;

        std::cout << "------------MOOK CHECKPOINT 1------\n";

        cv::copyMakeBorder(sample, sample, top, bottom, start, start - remainder_src , cv::BORDER_CONSTANT, 0);
        // cv::imshow("sample less than input network", sample);
        std::cout << "sample size: " << sample.size() << "\n";
    } 
    else
    {
        int start_index = (int) (sample_img.cols/2 - width/2);
        int end_index = (int) (sample_img.cols/2 + width/2);
        sample = sample_img(cv::Range::all(), cv::Range(start_index, end_index));
    }

    img_square = sample.clone();

    cv::Mat mSample;
    sample.convertTo(mSample, CV_32FC3);

    /* RESIZE RAW IMAGE FIRST TO GET ITS VARIATIONS */
    cv::Mat sample_resized;
    cv::Mat sample_resized_v2;
    cv::Mat sample_resized_v3;

    if(mSample.size() != input_geometry)
    {
        cv::resize(mSample, sample_resized, input_geometry);
    } 
    else 
    {
        sample_resized = mSample;
    }

    // cv::imshow("sample_resized", sample_resized);

    cv::Mat mSample_resized;

    // std::ofstream outx("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/sample_check/cpp/sample_resized_1.txt");
    // std::streambuf *coutbufx = std::cout.rdbuf();
    // std::cout.rdbuf(outx.rdbuf());
    //  std::cout << "mSample size: " << mSample.size() << "\n";
    //  std::cout << mSample << "\n";
    // std::cout.rdbuf(coutbufx);


    cv::resize(sample_resized, sample_resized_v2, cv::Size(0,0), 0.8, 0.8, cv::INTER_LINEAR);
    cv::resize(sample_resized, sample_resized_v3, cv::Size(0,0), 0.6, 0.6, cv::INTER_LINEAR);

    int length = sample_resized.rows;
    int half = std::floor(length/2);

    int length_v2 = sample_resized_v2.rows;
    int half_v2 = std::floor(length_v2/2);
    int remainder_v2 = length_v2 % 2;

    int start_v2 = (half - half_v2);
    int end_v2 = (half + half_v2) + remainder_v2;

    int length_v3 = sample_resized_v3.rows;
    int half_v3 = std::floor(length_v3/2);
    int remainder_v3 = length_v3 % 2;

    int start_v3 = (half - half_v3);
    int end_v3 = (half + half_v3) + remainder_v3;

    std::cout << "start_v2: " << start_v2 << "\n";
    std::cout << "start_v3: " << start_v3 << "\n";
    std::cout << "end_v2: " << end_v2 << "\n";
    std::cout << "end_v3: " << end_v3 << "\n";

    cv::Mat sample_resized_v2_padded;
    cv::Mat sample_resized_v3_padded;
    cv::copyMakeBorder(sample_resized_v2, sample_resized_v2_padded, start_v2, start_v2 - remainder_v2 , start_v2, start_v2 - remainder_v2 , cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
    cv::copyMakeBorder(sample_resized_v3, sample_resized_v3_padded, start_v3, start_v3 - remainder_v3 , start_v3, start_v3 - remainder_v3 , cv::BORDER_CONSTANT, cv::Scalar(128,128,128));



    cv::Mat offset_4 = cv::Mat(sample_resized_v2_padded.rows, sample_resized_v3_padded.cols, CV_32FC3, cv::Scalar(0.4,0.4,0.4));

    sample_resized = sample_resized.mul((1/255.0)*1.0) - offset_4;
    sample_resized_v2_padded = sample_resized_v2_padded.mul((1/255.0)*1.0) - offset_4;
    sample_resized_v3_padded = sample_resized_v3_padded.mul((1/255.0)*1.0) - offset_4;

    // cv::imshow("sample_resized_v2_padded", sample_resized_v2_padded);
    // cv::imshow("sample_resized_v3_padded", sample_resized_v3_padded);


    /* STORE IMAGE AT ALL SIZES */
    std::vector<cv::Mat> imgs;
    imgs.push_back(sample_resized);
    imgs.push_back(sample_resized_v2_padded);
    imgs.push_back(sample_resized_v3_padded);

    /* WRAP BATCH INPUT LAYER */
    std::vector<std::vector<cv::Mat>> input_batch;
    //got input_layer, width, height
    int num = input_layer->num(); //shape(0)
    float* input_data = input_layer->mutable_cpu_data();
    for(int j = 0; j < num; j++)
    {
        std::vector<cv::Mat> input_channels;
        for(int i = 0; i < input_layer->channels(); ++i)
        {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels.push_back(channel);
            input_data += width * height;
        }
        input_batch.push_back(std::vector<cv::Mat>(input_channels));
    }


    /* PREPROCESS BATCH */
    for(int i = 0; i < imgs.size(); i++)
    {
        // cv::Mat curr_img = imgs.at(i);
        cv::Mat curr_img;

        std::vector<cv::Mat> *input_chans = &(input_batch.at(i));

        cv::Mat curr_sample;
        curr_sample = curr_img;

        cv::Mat curr_sample_float;
        curr_sample.convertTo(curr_sample_float, CV_32FC3);

        cv::split(imgs.at(i), *input_chans);
        // cv::split(curr_sample_float, *input_chans);
    }

    std::cout << "----------GOING INTO NETWORK-----------\n";
/* --------------------------------------------------------------------------- */

    testnet->Forward();
    std::cout << "----------NETWORK HAS OUTPUT-----------\n";

    std::cout << "output_blobs.size()" << testnet->output_blobs().size() << "\n";

    //each one should have a size of 21, each 21 should have 46 x 46
    std::vector<std::vector<std::vector<float>>> heatmaps;
    std::vector<std::vector<std::vector<float>>> x_heatmaps;
    std::vector<std::vector<std::vector<float>>> y_heatmaps;
    std::vector<std::vector<std::vector<float>>> z_heatmaps;

    std::vector<int> heatmaps_h_w;
    std::vector<int> xhm_h_w;
    std::vector<int> yhm_h_w;
    std::vector<int> zhm_h_w;

    int outputblobi = 114;
    for (auto outputblob : testnet->output_blobs())
    {
        std::cout << "~~~~~~~~" << testnet->blob_names().at(outputblobi) << "\n";
        std::vector<std::vector<std::vector<float>>> curr_heatmaps;

        std::cout << "shape_string(): " << outputblob->shape_string() << "\n";

        for(int n = 0; n < outputblob->shape(0); n++)
        {
            std::vector<std::vector<float>> curr_num;
            for(int c = 0; c < outputblob->shape(1); c++)
            {
                std::vector<float> curr_channel;
                for(int h = 0; h < outputblob->shape(2); h++)
                {
                    for(int w = 0; w < outputblob->shape(3); w++)
                    {
                        curr_channel.push_back(outputblob->data_at(n,c,h,w));               
                    }
                }
                curr_num.push_back(curr_channel);
            }
            curr_heatmaps.push_back(curr_num);
        }

        switch(outputblobi)
        {
            case 114: 
                {
                    heatmaps = curr_heatmaps;
                    heatmaps_h_w = std::vector<int>(outputblob->shape().at(2), outputblob->shape().at(3));
                    break;
                }
            case 115: 
                {
                    x_heatmaps = curr_heatmaps; 
                    xhm_h_w = std::vector<int>(outputblob->shape().at(2), outputblob->shape().at(3));
                    break;
                }
            case 116: 
                {
                    y_heatmaps = curr_heatmaps;
                    yhm_h_w = std::vector<int>(outputblob->shape().at(2), outputblob->shape().at(3));
                    break;
                }
            case 117: 
                {
                    z_heatmaps = curr_heatmaps; 
                    zhm_h_w = std::vector<int>(outputblob->shape().at(2), outputblob->shape().at(3));
                    break;
                }
            default: std::cout << "OUTPUT OUT OF BOUND!!!"; break;
        }
        outputblobi++;

    }

    /* ------------- AVERAGE THE HEATMAPS OUT -------------  */
    int nums_out = testnet->output_blobs().at(0)->shape(0);
    int channels_out = testnet->output_blobs().at(0)->shape(1);
    int height_out = testnet->output_blobs().at(0)->shape(2);
    int width_out = testnet->output_blobs().at(0)->shape(3);

    std::vector<std::vector<cv::Mat>> h_sum;
    std::vector<std::vector<cv::Mat>> x_sum;
    std::vector<std::vector<cv::Mat>> y_sum;
    std::vector<std::vector<cv::Mat>> z_sum;

        /* -------- CONVERT HEATMAPS TO CV::MAT -------- */
    for(int n = 0; n < nums_out; n++)
    {
        std::vector<cv::Mat> hm_cv;
        std::vector<cv::Mat> xhm_cv;
        std::vector<cv::Mat> yhm_cv;
        std::vector<cv::Mat> zhm_cv;
        for(int c = 0; c < channels_out; c++)
        {
            cv::Mat hm_mat = cv::Mat(heatmaps_h_w.at(0), heatmaps_h_w.at(1), CV_32FC1);
            cv::Mat x_mat = cv::Mat(xhm_h_w.at(0), xhm_h_w.at(1), CV_32FC1);
            cv::Mat y_mat = cv::Mat(yhm_h_w.at(0), yhm_h_w.at(1), CV_32FC1);
            cv::Mat z_mat = cv::Mat(zhm_h_w.at(0), zhm_h_w.at(1), CV_32FC1);

            std::memcpy(hm_mat.data, heatmaps.at(n).at(c).data(), heatmaps.at(n).at(c).size()*sizeof(float));
            std::memcpy(x_mat.data, x_heatmaps.at(n).at(c).data(), x_heatmaps.at(n).at(c).size()*sizeof(float));
            std::memcpy(y_mat.data, y_heatmaps.at(n).at(c).data(), y_heatmaps.at(n).at(c).size()*sizeof(float));
            std::memcpy(z_mat.data, z_heatmaps.at(n).at(c).data(), z_heatmaps.at(n).at(c).size()*sizeof(float));

            hm_cv.push_back(hm_mat);
            xhm_cv.push_back(x_mat);
            yhm_cv.push_back(y_mat);
            zhm_cv.push_back(z_mat);    
        }
        h_sum.push_back(hm_cv);
        x_sum.push_back(xhm_cv);
        y_sum.push_back(yhm_cv);
        z_sum.push_back(zhm_cv);
    }

    /* ------RESIZE------*/
    std::vector<cv::Mat> hm_avg;
    std::vector<cv::Mat> xm_avg;
    std::vector<cv::Mat> ym_avg;
    std::vector<cv::Mat> zm_avg;

    cv::Size input_geometry_pool_scale = cv::Size(std::floor(input_geometry.width/8), std::floor(input_geometry.height/8)); 
    input_geometry_pool_scale = cv::Size(std::floor(input_geometry_pool_scale.width/2), std::floor(input_geometry_pool_scale.height/2));

    for(int c = 0; c < channels_out; c++)
    {
        // std::cout << "channel: " << c << "\n";
        cv::Mat hm_mat;
        cv::Mat hm_mat_prev;

        cv::Mat xm_mat;
        cv::Mat xm_mat_prev;

        cv::Mat ym_mat;
        cv::Mat ym_mat_prev;

        cv::Mat zm_mat;
        cv::Mat zm_mat_prev;
        for(int s = 0; s < scales_size; s++)
        {
            float rescale = 1.0/scales[s];
            cv::resize(h_sum.at(s).at(c), hm_mat, cv::Size(0,0), rescale, rescale, cv::INTER_LINEAR);
            cv::resize(x_sum.at(s).at(c), xm_mat, cv::Size(0,0), rescale, rescale, cv::INTER_LINEAR);
            cv::resize(y_sum.at(s).at(c), ym_mat, cv::Size(0,0), rescale, rescale, cv::INTER_LINEAR);
            cv::resize(z_sum.at(s).at(c), zm_mat, cv::Size(0,0), rescale, rescale, cv::INTER_LINEAR);

            // since square output, therefore use either cols or rows
            int start_index = std::floor(hm_mat.cols/2) - input_geometry_pool_scale.width; 
            int end_index = std::floor(hm_mat.cols/2) + input_geometry_pool_scale.width; 
            
            hm_mat = hm_mat(cv::Range(start_index, end_index), cv::Range(start_index, end_index)); //(rows, columns)
            xm_mat = xm_mat(cv::Range(start_index, end_index), cv::Range(start_index, end_index)); //(rows, columns)
            ym_mat = ym_mat(cv::Range(start_index, end_index), cv::Range(start_index, end_index)); //(rows, columns)
            zm_mat = zm_mat(cv::Range(start_index, end_index), cv::Range(start_index, end_index)); //(rows, columns)
            
            if(hm_mat_prev.empty()) {

                hm_mat_prev = hm_mat;
                xm_mat_prev = xm_mat;
                ym_mat_prev = ym_mat;
                zm_mat_prev = zm_mat;   
            } else {

                hm_mat += hm_mat_prev;
                hm_mat_prev = hm_mat;

                xm_mat += xm_mat_prev;
                xm_mat_prev = xm_mat;

                ym_mat += ym_mat_prev;
                ym_mat_prev = ym_mat;

                zm_mat += zm_mat_prev;
                zm_mat_prev = zm_mat;
            }
        }

        hm_mat /= scales_size;
        hm_avg.push_back(hm_mat);

        xm_mat /= scales_size;
        xm_avg.push_back(xm_mat);

        ym_mat /= scales_size;
        ym_avg.push_back(ym_mat);

        zm_mat /= scales_size;
        zm_avg.push_back(zm_mat);

    }

        /* ---- RESIZE ---- */
    std::vector<cv::Mat> hm_avg_input_size;
    for(int c = 0; c < channels_out; c++)
    {
        cv::Mat curr_hm;
        cv::resize(hm_avg.at(c), curr_hm, sample_resized.size(), 0, 0, cv::INTER_LINEAR);
        hm_avg_input_size.push_back(curr_hm);
    }

        /* ---- EXTRACT JOINT LOCATIONS FROM 2D HEATMAPS ---- */
    std::vector<cv::Point> max_locs;
    std::vector<cv::Point> joints_2d;
    for(auto hm : hm_avg_input_size)
    {
        double max;
        cv::Point max_loc;
        
        cv::minMaxLoc(hm, 0, &max, 0, &max_loc);
        max_locs.push_back(cv::Point(max_loc.x/8, max_loc.y/8));
        joints_2d.push_back(cv::Point(max_loc.x, max_loc.y));
    }

    std::vector<float> x_values;
    std::vector<float> y_values;
    std::vector<float> z_values;

    std::cout << "-------******* PRINT MAX_LOCS\n";
    for(int i = 0; i < max_locs.size(); i++)
    {
        x_values.push_back(xm_avg.at(i).at<float>(max_locs.at(i).y,max_locs.at(i).x) * 10);
        y_values.push_back(ym_avg.at(i).at<float>(max_locs.at(i).y,max_locs.at(i).x) * 10);
        z_values.push_back(zm_avg.at(i).at<float>(max_locs.at(i).y,max_locs.at(i).x) * 10);
    }

    std::vector<std::vector<float>> joints_3d_root_relative;// {{x0,y0,z0},{x1,y1,z1},{x2,y2,z2}...,{x20,y20,z20}}

    std::vector<float> root_relative_joints;
    root_relative_joints.push_back(x_values.at(14));
    root_relative_joints.push_back(y_values.at(14));
    root_relative_joints.push_back(z_values.at(14));

    for(int i = 0; i < max_locs.size(); i++)
    {
        float x = x_values.at(i);
        float y = y_values.at(i);
        float z = z_values.at(i);

        x -= root_relative_joints.at(0);
        y -= root_relative_joints.at(1);
        z -= root_relative_joints.at(2);

        std::vector<float> xyz;
        xyz.push_back(x); xyz.push_back(y); xyz.push_back(z);

        joints_3d_root_relative.push_back(xyz);
    }
    
    cv::Mat image_2d = img_square;

    for(int i = 0; i < joints_2d.size(); i++)
    {
        cv::circle(image_2d, joints_2d.at(i), 3, cv::Scalar(0,0,255), -1, 8);
    }


    // ------------- TO DRAW 2D POSE FROM CPP ---------------- //

    //---------------------------------SAVING 3D POINTS TO TEXT FILE---------------------------------//

                                /*---- PRINT VNECT_3D_JOINTS.TXT ----*/
                    std::string saveToName = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/3d_joints/" + croppedImgName + ".txt";
                    std::ofstream out3djoints(saveToName);
                    std::streambuf *coutbuf3djoints = std::cout.rdbuf();
                    std::cout.rdbuf(out3djoints.rdbuf());
                    
                    // int joint_i = 0;
                    for(auto joint : joints_3d_root_relative)
                    {
                        // if(joint_i == 15) break;
                        std::cout << joint.at(0) << " ";
                        std::cout << joint.at(1) << " ";
                        std::cout << joint.at(2) << " ";
                        std::cout << "\n";
                        // joint_i++;
                    }
                    
                    std::cout.rdbuf(coutbuf3djoints);
                    std::cout << "written to vnect_3d_joints.txt\n";

        /* ---- display images ---- */
    cv::imshow(croppedImgName, image_2d);
    // cv::imshow("img", img);
    // cv::imwrite("./outputs/vnect/frame2238/image_2d.jpg", image_2d);
    // cv::waitKey(0);

    std::cout << "----------done----------\n";
}

void mCropPose(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr)
{
    try 
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
// ----------------------------------------------------------------------------------- //
            auto& poseKeypoints = datumsPtr->at(0)->poseKeypoints;
            std::cout << "poseKeypoints getSize size: " << poseKeypoints.getSize().size() << "\n";
            std::cout << "poseKeypoints getSize 0: " << poseKeypoints.getSize(0) << "\n";
            std::cout << "poseKeypoints getSize 1: " << poseKeypoints.getSize(1) << "\n";
            std::cout << "poseKeypoints getSize 2: " << poseKeypoints.getSize(2) << "\n";

            /////////////////////////////////////////////////////////////////////////////////////////////////////
            //////////SET STD::VECTOR<STD::VECTOR<CV::POINT>>////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////

            std::vector<std::vector<float>> poses;
            int numPoses = poseKeypoints.getSize(0);
            int numEachPose = poseKeypoints.getSize(1)*poseKeypoints.getSize(2);

            std::cout << "numPoses: " << numPoses << "\n";
            std::cout << "numEachPose: " << numEachPose << "\n";

            for(int pose = 0; pose < numPoses; pose++)
            {
                // std::cout << "----------POSE " << pose+1 << "\n";
                std::vector<float> posePoints_x;
                std::vector<float> posePoints_y;
                std::cout << "pose: " << pose << "\n";
                std::cout << "(pose+1)*numEachPose: " << (pose+1)*numEachPose << "\n";
                for(int j = pose*numEachPose; j < (pose+1)*numEachPose; j += 3)
                {
                    // std::cout << poseKeypoints.at(j) << "\n";
                    posePoints_x.push_back(poseKeypoints.at(j));
                    posePoints_y.push_back(poseKeypoints.at(j+1));
                }
                poses.push_back(posePoints_x);
                poses.push_back(posePoints_y);
            }


            std::vector<std::vector<float>> posesBounds;
            for(int i = 0; i < poses.size(); i += 2)
            {
                // std::cout << "           " << i << "\n";
                // std::cout << "poses.at(i).size(): " << poses.at(i).size() << "\n";
                // std::cout << "min element: " << findMin(poses.at(i)) << "\n";
                // std::cout << "max element: " << findMax(poses.at(i)) << "\n";

                // std::cout << "           " << i+1 << "\n";
                // std::cout << "poses.at(i+1).size(): " << poses.at(i+1).size() << "\n";
                // std::cout << "min element: " << findMin(poses.at(i+1)) << "\n";
                // std::cout << "max element: " << findMax(poses.at(i+1)) << "\n";

                std::vector<float> bounds;
                bounds.push_back(findMin(poses.at(i))); //x //need to deal with 0.00 ?
                bounds.push_back(findMax(poses.at(i))); //x //need to deal with 0.00 ?
                bounds.push_back(findMin(poses.at(i+1))); //y //need to deal with 0.00 ?
                bounds.push_back(findMax(poses.at(i+1))); //y //need to deal with 0.00 ?
                posesBounds.push_back(bounds);
            }

            cv::Mat image = datumsPtr->at(0)->cvInputData;

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


                vnect(croppedImg, vname);
            }
            /////////////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////
            /////////////////////////////////////////////////////////////////////////////////////////////////////

            // cv::imshow("image from mCrop", image);

            std::cout << "---------------------------posesBounds size: " << posesBounds.size() << "\n";
            std::cout << "finding distances between poses == Array should contain poses - 1 differences.\n";
            std::vector<float> poseDistances;

            for (int person = 0 ; person < poseKeypoints.getSize(0) - 1 ; person++)
            {
                std::cout << "poseKeypoints[{person, 1, 0}]: " << poseKeypoints[{person, 1, 0}] << "\n";
                std::cout << "poseKeypoints[{person+1, 1, 0}]: " << poseKeypoints[{person+1, 1, 0}] << "\n";
                std::cout << "poseKeypoints[{person, 14, 0}]: " << poseKeypoints[{person, 14, 0}] << "\n";
                std::cout << "poseKeypoints[{person+1, 14, 0}]: " << poseKeypoints[{person+1, 14, 0}] << "\n";
                float neckDiff = poseKeypoints[{person+1, 1, 0}] - poseKeypoints[{person, 1, 0}];
                float chestDiff = poseKeypoints[{person+1, 14, 0}] - poseKeypoints[{person, 14, 0}];

                std::cout << "person i [13]: " << poseKeypoints[{person, 13, 1}] << ", [10]: " <<  poseKeypoints[{person, 10, 1}] << "\n";
                std::cout << "person i+1 [13]: " << poseKeypoints[{person+1, 13, 1}] << ", [10]: " <<  poseKeypoints[{person+1, 10, 1}] << "\n";
                float lowestAnkle_1 = poseKeypoints[{person, 13, 1}] > poseKeypoints[{person, 10, 1}] ? poseKeypoints[{person, 13, 1}] : poseKeypoints[{person, 10, 1}];  // y values of {LAnkle: 13}, {RAnkel: 10}
                float lowestAnkle_2 = poseKeypoints[{person+1, 13, 1}] > poseKeypoints[{person+1, 10, 1}] ? poseKeypoints[{person+1, 13, 1}] : poseKeypoints[{person+1, 10, 1}];
                std::cout << "lowestAnkle_1: " << lowestAnkle_1 << "\n";
                std::cout << "lowestAnkle_2: " << lowestAnkle_2 << "\n";

                float ankleRatio = lowestAnkle_1 > lowestAnkle_2 ? (lowestAnkle_1 - lowestAnkle_2) : (lowestAnkle_2 - lowestAnkle_1);
                std::cout << "ankleRatio: " << ankleRatio << ", 1/ankleRatio: " << (1/ankleRatio) << "\n";

                std::cout << "neckDiff: " << neckDiff << "\n";
                std::cout << "chestDiff: " << chestDiff << "\n";

                // poseDistances.push_back(neckDiff);
            }


            std::vector<std::vector<float>> jointsRad;
            std::vector<std::vector<cv::Point>> jointsCen;

            cv::Mat currPose = datumsPtr->at(0)->cvOutputData.clone();
            cv::Mat currPose_output = datumsPtr->at(0)->cvOutputData.clone();
            float opacity = 0.2;
            for(int i = 0; i < poseKeypoints.getSize(0); i++)
            {

                std::vector<float> currRad;
                std::vector<cv::Point> currCen;


                float x_neck = poseKeypoints[{i,1,0}];
                float y_neck = poseKeypoints[{i,1,1}];

                float x_head = poseKeypoints[{i,0,0}];
                float y_head = poseKeypoints[{i,0,1}];

                float x_chest = poseKeypoints[{i,14,0}];
                float y_chest = poseKeypoints[{i,14,1}];

                float l_head = pow(pow((x_head - x_neck),2) + pow((y_head - y_neck),2), 0.5);
                float l_chest = pow(pow((x_chest - x_neck),2) + pow((y_chest - y_neck),2), 0.5);

                float radius = l_head > l_chest ? l_head : l_chest;

                std::cout << "neck radius: " << radius << "\n";
                currRad.push_back(radius);

                cv::Point center = {(int)x_neck,(int)y_neck};
                currCen.push_back(center);

                // int r = 10;
                cv::circle(currPose, center, radius, cv::Scalar(0,0,255), -1, 8);
                cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);



                float x_lhip = poseKeypoints[{i,11,0}];
                float y_lhip = poseKeypoints[{i,11,1}];
                float x_lankle = poseKeypoints[{i,13,0}];
                float y_lankle = poseKeypoints[{i,13,1}];

                float x_rhip = poseKeypoints[{i,8,0}];
                float y_rhip = poseKeypoints[{i,8,1}];
                float x_rankle = poseKeypoints[{i,10,0}];
                float y_rankle = poseKeypoints[{i,10,1}];

                float x_lknee = poseKeypoints[{i,12,0}];
                float y_lknee = poseKeypoints[{i,12,1}];

                float x_rknee = poseKeypoints[{i,9,0}];
                float y_rknee = poseKeypoints[{i,9,1}];

                float l_lhip = pow(pow((x_lhip - x_lknee),2) + pow((y_lhip - y_lknee),2), 0.5);
                float l_lankle = pow(pow((x_lankle - x_lknee),2) + pow((y_lankle - y_lknee),2), 0.5);

                float l_rhip = pow(pow((x_rhip - x_rknee),2) + pow((y_rhip - y_rknee),2), 0.5);
                float l_rankle = pow(pow((x_rankle - x_rknee),2) + pow((y_rankle - y_rknee),2), 0.5);

                radius = l_lhip > l_lankle ? l_lhip : l_lankle;

                std::cout << "lhip radius: " << radius << "\n";
                currRad.push_back(radius);

                center = {(int)x_lknee, (int)y_lknee};
                currCen.push_back(center);

                if(x_lknee != 0 && y_lknee != 0)
                {
                    cv::circle(currPose, center, radius, cv::Scalar(0,255,0), -1, 8);
                    cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);
                }

                radius = l_rhip > l_rankle ? l_rhip : l_rankle;
                
                std::cout << "rhip radius: " << radius << "\n";
                currRad.push_back(radius);

                center = {(int)x_rknee, (int)y_rknee};
                currCen.push_back(center);

                // std::cout << "x_rknee: " << x_rknee << "\n";
                // std::cout << "y_rknee: " << y_rknee << "\n";
                // if(x_rknee != 0 && y_rknee != 0)
                // {
                    cv::circle(currPose, center, radius, cv::Scalar(255,0,0), -1, 8);
                    cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
                // }

                float diameter = pow(pow((x_chest - x_lhip),2) + pow((y_chest - y_lhip),2), 0.5); // diameter of chest/lhip
                if(x_chest == 0 || x_lhip == 0 || y_chest == 0 || y_lhip == 0)
                {
                    diameter = 0;
                }
                radius = diameter/2;
                currRad.push_back(radius);

                std::cout << "......currRad chest/lhip: " << radius << "\n";
                std::cout << "x_chest: " << x_chest << ", x_lhip: " << x_lhip << "\n";
                std::cout << "y_chest: " << y_chest << ", y_lhip: " << y_lhip << "\n";

                center = {(int)(x_chest + x_lhip)/2, (int)(y_chest + y_lhip)/2};
                currCen.push_back(center);

                if(x_chest != 0 && y_chest != 0 && x_lhip != 0 && y_lhip != 0)
                {
                    cv::circle(currPose, center, radius, cv::Scalar(255,0,0), -1, 8);
                    cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
                }

                diameter = pow(pow((x_chest - x_rhip),2) + pow((y_chest - y_rhip),2), 0.5); // diameter of chest/rhip
                radius = diameter/2;
                currRad.push_back(radius);

                std::cout << "......currRad chest/rhip: " << radius << "\n";
                std::cout << "x_chest: " << x_chest << ", x_rhip: " << x_rhip << "\n";
                std::cout << "y_chest: " << y_chest << ", y_rhip: " << y_rhip << "\n";

                center = {(int)(x_chest + x_rhip)/2, (int)(y_chest + y_rhip)/2};
                currCen.push_back(center);

                if(x_chest != 0 && y_chest != 0 && x_rhip != 0 && y_rhip != 0)
                {
                    // std::cout << "!!!!!!!!!!!!!!!!RENDERING CHECK RHIP!!!!!!!!!!!!!!!!\n";
                    cv::circle(currPose, center, radius, cv::Scalar(255,0,0), -1, 8);
                    cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
                }
                // break;                

                jointsRad.push_back(currRad);
                jointsCen.push_back(currCen);

            }

            std::cout << "------------------------------------------\n\n";

            std::cout << "jointsRad size: " << jointsRad.size() << "\n";
            std::cout << "jointsCen size: " << jointsCen.size() << "\n";

            std::cout << "------------------------------------------\n\n";

            std::cout << "-----:::: jointsRad ::::-----\n";
            for(int i = 0; i < jointsRad.size(); i++)
            {
                std::cout << "jointsRad i = " << i << "\n";
                for(int j = 0; j < jointsRad.at(i).size(); j++)
                {
                    std::cout << "  " << j << ". " << jointsRad.at(i).at(j) << "\n";
                }
            }

            std::cout << "-----:::: jointsCen ::::-----\n";
            for(int i = 0; i < jointsCen.size(); i++)
            {
                std::cout << "jointsCen i = " << i << "\n";
                for(int j = 0; j < jointsCen.at(i).size(); j++)
                {
                    std::cout << "  " << j << ". " << jointsCen.at(i).at(j) << "\n";
                }
            }


            std::vector<std::vector<cv::Point>> jointsCenExtr;

            std::cout << "-----:::: find extreme points at head, floor left and floor right ::::-----\n";
            for(int i = 0; i < jointsRad.size(); i++)
            {
                std::vector<cv::Point> currExtr;
                for(int j = 0; j < jointsRad.at(i).size(); j++)
                {   
                    if(j == 0) //neck
                    {
                        // float neck_top_x = jointsCen.at(i).at(j).x + jointsRad.at(i).at(j);
                        float neck_top_y = jointsCen.at(i).at(j).y - jointsRad.at(i).at(j);
                        currExtr.push_back(cv::Point(jointsCen.at(i).at(j).x, neck_top_y));
                    }
                    else if(j == 1)
                    {
                        // float lknee_bottom_x = jointsCen.at(i).at(j).x + jointsRad.at(i).at(j);
                        float lknee_bottom_y = jointsCen.at(i).at(j).y + jointsRad.at(i).at(j);
                        currExtr.push_back(cv::Point(jointsCen.at(i).at(j).x, lknee_bottom_y));
                    }
                    else if(j == 2)
                    {
                        float rknee_bottom_y = jointsCen.at(i).at(j).y + jointsRad.at(i).at(j);
                        currExtr.push_back(cv::Point(jointsCen.at(i).at(j).x, rknee_bottom_y));
                    }
                }
                jointsCenExtr.push_back(currExtr);
            }

            for(int i = 0; i < jointsCenExtr.size(); i++)
            {
                std::cout << "  i: " << i << "\n";


                std::cout << "neck y: " << jointsCenExtr.at(i).at(0).y << "\n";
                std::cout << "lhip y: " << jointsCenExtr.at(i).at(1).y << "\n";
                std::cout << "rhip y: " << jointsCenExtr.at(i).at(2).y << "\n";

                int maxHeight = jointsCenExtr.at(i).at(1).y > jointsCenExtr.at(i).at(2).y ? abs(jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(0).y) : abs(jointsCenExtr.at(i).at(2).y - jointsCenExtr.at(i).at(0).y);

                std::cout << "jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(0).y ==> " << abs(jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(0).y) << "\n";
                std::cout << "jointsCenExtr.at(i).at(2).y - jointsCenExtr.at(i).at(0).y ==> " << abs(jointsCenExtr.at(i).at(2).y - jointsCenExtr.at(i).at(0).y) << "\n";
                std::cout << "maxHeight: " << maxHeight << "\n";

                std::cout << "pose 0 lowest hip: " << ((jointsCenExtr.at(0).at(1).y > jointsCenExtr.at(0).at(2).y) ? jointsCenExtr.at(0).at(1).y : jointsCenExtr.at(0).at(2).y) << "\n";
                std::cout << "pose 1 lowest hip: " << ((jointsCenExtr.at(1).at(1).y > jointsCenExtr.at(1).at(2).y) ? jointsCenExtr.at(1).at(1).y : jointsCenExtr.at(1).at(2).y) << "\n";

                // std::cout << "jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(2).y ==> " << jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(2).y << "\n";

                for(int j = 0; j < jointsCenExtr.at(i).size()-3; j++)
                {
                    std::cout << j << ". " << jointsCenExtr.at(i).at(j) << "\n";
                    cv::circle(currPose_output, jointsCenExtr.at(i).at(j), 5, cv::Scalar(0,255,255), -1, 8);

                    // find max length in height


                }
            }


            std::cout << "------------- CHECK JOINTS RAD AND CEN -------------\n";
            std::vector<std::vector<float>> jointsChose;
            std::vector<std::vector<cv::Point>> jointsChosePoints;

            for(int i = 0; i < jointsRad.size(); i++)
            {
                std::cout << "==i: " << i << "\n";
                
                std::vector<float> choPart;
                std::vector<cv::Point> choPoints;

                float top = jointsRad.at(i).at(0) ;
                float bot = jointsRad.at(i).at(1) > jointsRad.at(i).at(2) ? jointsRad.at(i).at(1) : jointsRad.at(i).at(2);
                float mid = jointsRad.at(i).at(3) > jointsRad.at(i).at(4) ? jointsRad.at(i).at(3) : jointsRad.at(i).at(4);

                cv::Point topPoint = jointsCen.at(i).at(0);
                cv::Point botPoint = jointsRad.at(i).at(1) > jointsRad.at(i).at(2) ? jointsCen.at(i).at(1) : jointsCen.at(i).at(2);
                cv::Point midPoint = jointsRad.at(i).at(3) > jointsRad.at(i).at(4) ? jointsCen.at(i).at(3) : jointsCen.at(i).at(4);

                choPart.push_back(top);
                choPart.push_back(bot);
                choPart.push_back(mid);

                choPoints.push_back(topPoint);
                choPoints.push_back(botPoint);
                choPoints.push_back(midPoint);
                
                jointsChose.push_back(choPart);
                jointsChosePoints.push_back(choPoints);

                std::cout << "top: " << top << "\n";
                std::cout << "bot: " << bot << "\n";
                std::cout << "mid: " << mid << "\n";

                std::cout << "topPoint: " << topPoint << "\n";
                std::cout << "botPoint: " << botPoint << "\n";
                std::cout << "midPoint: " << midPoint << "\n";



                for(int j = 0; j < jointsRad.at(i).size(); j++)
                {
                    std::cout << j << ". " << jointsRad.at(i).at(j) << "\n";

                }
            }

            std::cout << "jointsChose size: " << jointsChose.size() << "\n";
            std::cout << "jointsChosePoints size: " << jointsChosePoints.size() << "\n";

            std::cout << "------------- JOINTS AVG -------------\n";
            std::vector<float> poseFloorLevel; 
            for(int i = 0; i < jointsChose.size(); i++)
            {
                std::cout << "i: " << i << "\n";
                std::cout << "floor: " << jointsChosePoints.at(i).at(1).y + jointsChose.at(i).at(1) << "\n";
                float floorLevel = jointsChosePoints.at(i).at(1).y + jointsChose.at(i).at(1);
                poseFloorLevel.push_back(floorLevel);
                for(int j = 0; j < jointsChose.at(i).size(); j++)
                {
                    std::cout << jointsChose.at(i).at(j) << "\n";
                    std::cout << jointsChosePoints.at(i).at(j) << "\n";
                }
            }

            // for now, kind of assuming every pose has the same height or whatever height given by VNect
            for(int i = 0; i < poseFloorLevel.size()-1; i++)
            {
                std::cout << "floor level difference: " << (float) abs(poseFloorLevel.at(i) - poseFloorLevel.at(i+1)) << "\n";
            }
            // compare radius
            // depth depends on poses interdependency positions mainly, not where there are in the original image




            // cv::imshow("currPose_output", currPose_output);
            // cv::imwrite("media_2_proportion_pose.png", currPose_output);

            // op::log("Body keypoints: " + datumsPtr->at(0)->poseKeypoints.toString(), op::Priority::High);

// ----------------------------------------------------------------------------------- //
            PyObject* pInt;
            char filename[] = "../drawMultiPerson.py";
            FILE* fp;
            Py_Initialize();
            // PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
            fp = fopen(filename, "r");
            PyRun_SimpleFile(fp, filename);    
            Py_Finalize();

            // wchar_t *program = Py_DecodeLocale("../drawMultiPerson.py", NULL);
            // if (program == NULL) {
            //     fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
            //     exit(1);
            // }
            // Py_SetProgramName(program);
            // Py_Initialize();
            // if (Py_FinalizeEx() < 0) {
            //     exit(120);
            // }
            // PyMem_RawFree(program);


        }
    }
    catch (const std::exception& e)
    {
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        std::cout << "in exception mPoseKeyPoints\n";
    }
}

int tutorialApiCpp()
{
    try
    {   
        // openLogFile();
        // printToFile("Welcome to Sanrio Land");
        // freopen("SavemetheLastDance.txt","w",stdout);
        op::log("----------------------------------------Starting OpenPose demo...", op::Priority::High);

        const auto opTimer = op::getTimerInit();

        // Required flags to enable heatmaps
        FLAGS_heatmaps_add_parts = true;
        FLAGS_heatmaps_add_bkg = true;
        FLAGS_heatmaps_add_PAFs = true;
        FLAGS_heatmaps_scale = 2;

        FLAGS_part_candidates = true;

        // Configuring OpenPose
        op::log("----------------------------------------Configuring OpenPose...", op::Priority::High);
        op::Wrapper opWrapper{op::ThreadManagerMode::Asynchronous};
        // std::cout << "--------opWrapper{op::ThreadManagerMode::Asynchronous} done,, about to configureWrapper(opWrapper)\n";
        configureWrapper(opWrapper);
        
        // Starting OpenPose
        op::log("----------------------------------------Starting thread(s)...", op::Priority::High);
        opWrapper.start();

        std::cout << "----------------------------------------opWrapper.started()" << "\n";

        // Process and display image
        const auto imageToProcess = cv::imread(FLAGS_image_path);

        // std::cout << "imageToProcess size: " << imageToProcess.size() << "\n";
        std::cout << "----------------------------------------cv imread image path" << "\n";
        auto datumProcessed = opWrapper.emplaceAndPop(imageToProcess);

        std::cout << "----------------------------------------opWrapper emplaced and popped" << "\n";
        if (datumProcessed != nullptr)
        {
            /*--------------- GET HEATMAPS IN ANY GIVEN OUTPUT FORM ---------------*/
            // const auto& poseHMs = datumProcessed->at(0)->poseHeatMaps;
            // std::cout << "*********Mook log:: poseHMs: " << poseHMs << "\n";
            /*--------------- END GET HEATMAPS IN ANY GIVEN OUTPUT FORM ---------------*/

            // printKeypoints(datumProcessed);
            if (!FLAGS_no_display)
            {
                // mPoseKeyPoints(datumProcessed);
                // mOutputData(datumProcessed);
                // mPoseCandidates(datumProcessed);
                mCropPose(datumProcessed);
                display(datumProcessed);
                const auto numberChannels = datumProcessed->at(0)->poseHeatMaps.getSize(0);
                // std::cout << "****^^Mookie the Rookie Cookie Smoothie presents^^" << "\n";
                // std::cout << "---------datumProcessed->at(0)->poseHeatMaps.getSize(0): " << datumProcessed->at(0)->poseHeatMaps.getSize(0) << "\n";
                // std::cout << "---------datumProcessed->at(0)->poseHeatMaps.getSize(1): " << datumProcessed->at(0)->poseHeatMaps.getSize(1) << "\n";
                // std::cout << "---------datumProcessed->at(0)->poseHeatMaps.getSize(2): " << datumProcessed->at(0)->poseHeatMaps.getSize(2) << "\n";
                
                // for (auto desiredChannel = 0; desiredChannel < numberChannels; desiredChannel)
                // {
                //     if(displayHeatMaps(datumProcessed, desiredChannel))
                //     {
                //         std::cout << "----------------------------------------displayHeatMaps(datumProcessed, desiredChannel)" << "\n";
                //         break;
                //     }    
                // }                
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
        // closeLogFile();
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

