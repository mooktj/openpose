#ifdef USE_CAFFE
    #include <atomic>
    #include <mutex>
    #include <caffe/net.hpp>
    #include <glog/logging.h> // google::InitGoogleLogging
#endif

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// #include <caffe/caffe.hpp>
#include <openpose/core/common.hpp>
#include <openpose/vnect/vNect.hpp>
#include <openpose/vnect/draw3DPython.hpp>

namespace op
{
   	float vNectFindMax(std::vector<float> in)
    {
        float max = 0;
        for(unsigned int i = 0; i < in.size(); i++)
        {
            if(max < in.at(i)) max = in.at(i);
        }
        return max;
    }

    float vNectFindMin(std::vector<float> in)
    {
        float min = 1000000;
        for(unsigned int i = 0; i < in.size(); i++)
        {
            if(min > in.at(i)) min = in.at(i);
        }
        return min;
    }

    void vNectPostForward(const std::shared_ptr<op::Datum>& datumsPtr)
    {
    	auto& poseKeypoints = datumsPtr->poseKeypoints;
    	std::vector<float> poseDistances;

    	// for (int person = 0 ; person < poseKeypoints.getSize(0) - 1 ; person++)
     //    {
     //        float neckDiff = poseKeypoints[{person+1, 1, 0}] - poseKeypoints[{person, 1, 0}];
     //        float chestDiff = poseKeypoints[{person+1, 14, 0}] - poseKeypoints[{person, 14, 0}];

     //        float lowestAnkle_1 = poseKeypoints[{person, 13, 1}] > poseKeypoints[{person, 10, 1}] ? poseKeypoints[{person, 13, 1}] : poseKeypoints[{person, 10, 1}];  // y values of {LAnkle: 13}, {RAnkel: 10}
     //        float lowestAnkle_2 = poseKeypoints[{person+1, 13, 1}] > poseKeypoints[{person+1, 10, 1}] ? poseKeypoints[{person+1, 13, 1}] : poseKeypoints[{person+1, 10, 1}];

     //        float ankleRatio = lowestAnkle_1 > lowestAnkle_2 ? (lowestAnkle_1 - lowestAnkle_2) : (lowestAnkle_2 - lowestAnkle_1);

     //        // poseDistances.push_back(neckDiff);
     //    }

		std::vector<std::vector<float>> jointsRad;
		std::vector<std::vector<cv::Point>> jointsCen;

		cv::Mat currPose = datumsPtr->cvOutputData.clone();
		cv::Mat currPose_output = datumsPtr->cvOutputData.clone();
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

            currRad.push_back(radius);

            center = {(int)x_lknee, (int)y_lknee};
            currCen.push_back(center);

            if(x_lknee != 0 && y_lknee != 0)
            {
                cv::circle(currPose, center, radius, cv::Scalar(0,255,0), -1, 8);
                cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);
            }

            radius = l_rhip > l_rankle ? l_rhip : l_rankle;
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

            center = {(int)(x_chest + x_rhip)/2, (int)(y_chest + y_rhip)/2};
            currCen.push_back(center);

            if(x_chest != 0 && y_chest != 0 && x_rhip != 0 && y_rhip != 0)
            {
                cv::circle(currPose, center, radius, cv::Scalar(255,0,0), -1, 8);
                cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            }
            // break;                

            jointsRad.push_back(currRad);
            jointsCen.push_back(currCen);

        }

        // "-----:::: jointsRad ::::-----" //
        for(int i = 0; i < jointsRad.size(); i++)
        {
            std::cout << "jointsRad i = " << i << "\n";
            for(int j = 0; j < jointsRad.at(i).size(); j++)
            {
                std::cout << "  " << j << ". " << jointsRad.at(i).at(j) << "\n";
            }
        }

        // "-----:::: jointsCen ::::-----" //
        for(int i = 0; i < jointsCen.size(); i++)
        {
            std::cout << "jointsCen i = " << i << "\n";
            for(int j = 0; j < jointsCen.at(i).size(); j++)
            {
                std::cout << "  " << j << ". " << jointsCen.at(i).at(j) << "\n";
            }
        }

        std::vector<std::vector<cv::Point>> jointsCenExtr;

        // "-----:::: find extreme points at head, floor left and floor right ::::-----" //
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
            int maxHeight = jointsCenExtr.at(i).at(1).y > jointsCenExtr.at(i).at(2).y ? abs(jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(0).y) : abs(jointsCenExtr.at(i).at(2).y - jointsCenExtr.at(i).at(0).y);
            // std::cout << "jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(2).y ==> " << jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(2).y << "\n";
            for(int j = 0; j < jointsCenExtr.at(i).size()-3; j++)
            {
                // std::cout << j << ". " << jointsCenExtr.at(i).at(j) << "\n";
                cv::circle(currPose_output, jointsCenExtr.at(i).at(j), 5, cv::Scalar(0,255,255), -1, 8);
                // find max length in height
            }
        }
	
		// "------------- CHECK JOINTS RAD AND CEN -------------" //
        std::vector<std::vector<float>> jointsChose;
        std::vector<std::vector<cv::Point>> jointsChosePoints;
		for(int i = 0; i < jointsRad.size(); i++)
        {            
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
        }        

        // "------------- JOINTS AVG -------------" //
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

		cv::imshow("currPose_output", currPose_output);


    }


    void vNectForward(cv::Mat croppedImg, std::string croppedImgName, std::string pathToWrite)
    {
    	caffe::Caffe::set_mode(caffe::Caffe::GPU);

    	const std::string modelFolder = "./models/";
	    const std::string protoTxtfile = modelFolder + "vnect/vnect_net.prototxt";
	    const std::string trainedModelFile = modelFolder + "vnect/vnect_model.caffemodel";

	    float scales[3] = {1.0, 0.8, 0.6};
	    int scales_size = sizeof(scales)/sizeof(scales[0]);

	    std::shared_ptr<caffe::Net<float>> testnet;
	    testnet.reset(new caffe::Net<float>(protoTxtfile, caffe::TEST));
	    testnet->CopyTrainedLayersFrom(trainedModelFile);

	    caffe::Blob<float>* input_layer = testnet->input_blobs()[0];

	    int num_channels = input_layer->channels();

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

	    cv::Mat sample_resized_v2_padded;
	    cv::Mat sample_resized_v3_padded;
	    cv::copyMakeBorder(sample_resized_v2, sample_resized_v2_padded, start_v2, start_v2 - remainder_v2 , start_v2, start_v2 - remainder_v2 , cv::BORDER_CONSTANT, cv::Scalar(128,128,128));
	    cv::copyMakeBorder(sample_resized_v3, sample_resized_v3_padded, start_v3, start_v3 - remainder_v3 , start_v3, start_v3 - remainder_v3 , cv::BORDER_CONSTANT, cv::Scalar(128,128,128));

	    cv::Mat offset_4 = cv::Mat(sample_resized_v2_padded.rows, sample_resized_v3_padded.cols, CV_32FC3, cv::Scalar(0.4,0.4,0.4));

	    sample_resized = sample_resized.mul((1/255.0)*1.0) - offset_4;
	    sample_resized_v2_padded = sample_resized_v2_padded.mul((1/255.0)*1.0) - offset_4;
	    sample_resized_v3_padded = sample_resized_v3_padded.mul((1/255.0)*1.0) - offset_4;

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

	    testnet->Forward();

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
	        std::vector<std::vector<std::vector<float>>> curr_heatmaps;

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

	    cv::imshow(croppedImgName, image_2d);

		/*---- PRINT VNECT_3D_JOINTS.TXT ----*/
		// "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/3d_joints/"
		std::string saveToName = pathToWrite + croppedImgName + ".txt";
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

		// ----- TESTING draw3DPython ----- //
		std::vector<std::string> fileNames;
		draw3DPython(fileNames);
    }
}