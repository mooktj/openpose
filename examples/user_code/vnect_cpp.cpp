#include <caffe/caffe.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <string>
#include <cstdio>
#include <fstream>
#include <cstring>

#include <math.h>

int main(int argc, char *argv[])
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
	// std::cout << "input_layer->width() = " << input_layer->width() << "\n";
	// std::cout << "input_layer->height() = " << input_layer->height() << "\n";

	// const std::string file = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/vnect_media/frame899.jpg";
	// const std::string file = "/home/mooktj/Desktop/myworkspace/img2vid/kim_yuna/2013-FS-Practice/frame2238.jpg";
	// const std::string file = "/home/mooktj/Desktop/myworkspace/img2vid/kim_yuna/Mook-VNect/data/mpii_3dhp_ts6/cam5_frame000131.jpg";
	// const std::string file = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/COCO_val2014_000000000192.jpg";
	// const std::string file = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_2.png";
	const std::string file = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/examples/media/multi-person/media_2_2.png";

	cv::Mat img = cv::imread(file);

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
		cv::imshow("sample less than input network", sample);
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

		// std::ofstream out1("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/sample_check/cpp/mSample.txt");
		// std::streambuf *coutbuf1 = std::cout.rdbuf();
		// std::cout.rdbuf(out1.rdbuf());
		// 	std::cout << "mSample size: " << mSample.size() << "\n";
		// 	std::cout << mSample << "\n";
		// std::cout.rdbuf(coutbuf1);

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

	cv::imshow("sample_resized", sample_resized);

	cv::Mat mSample_resized;

	// std::ofstream outx("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/sample_check/cpp/sample_resized_1.txt");
	// std::streambuf *coutbufx = std::cout.rdbuf();
	// std::cout.rdbuf(outx.rdbuf());
	// 	std::cout << "mSample size: " << mSample.size() << "\n";
	// 	std::cout << mSample << "\n";
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

	cv::imshow("sample_resized_v2_padded", sample_resized_v2_padded);
	cv::imshow("sample_resized_v3_padded", sample_resized_v3_padded);


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

	// max_locs.clear();
	// max_locs.push_back(cv::Point(179.0, 36.0));
	// max_locs.push_back(cv::Point(190.418823, 75.025108));
	// max_locs.push_back(cv::Point(169.444, 87.391144));
	// max_locs.push_back(cv::Point(134.257537, 96.957237));
	// max_locs.push_back(cv::Point(96.157257, 98.822632));
	// max_locs.push_back(cv::Point(218.044708, 81.698891));
	// max_locs.push_back(cv::Point(262.786651, 106.418678));
	// max_locs.push_back(cv::Point(254.220245, 150.230949));
	// max_locs.push_back(cv::Point(204.696167, 163.569206));
	// max_locs.push_back(cv::Point(174.231231, 213.079903));
	// max_locs.push_back(cv::Point(183.740234, 278.808601));
	// max_locs.push_back(cv::Point(233.289581, 158.803978));
	// max_locs.push_back(cv::Point(242.771667, 225.459358));
	// max_locs.push_back(cv::Point(281.831085, 277.826485));
	// max_locs.push_back(cv::Point(205.658874, 122.621582));
	// // 
	// joints_2d.clear();
	// joints_2d.push_back(cv::Point(179.0, 36.0));
	// joints_2d.push_back(cv::Point(190.418823, 75.025108));
	// joints_2d.push_back(cv::Point(169.444, 87.391144));
	// joints_2d.push_back(cv::Point(134.257537, 96.957237));
	// joints_2d.push_back(cv::Point(96.157257, 98.822632));
	// joints_2d.push_back(cv::Point(218.044708, 81.698891));
	// joints_2d.push_back(cv::Point(262.786651, 106.418678));
	// joints_2d.push_back(cv::Point(254.220245, 150.230949));
	// joints_2d.push_back(cv::Point(204.696167, 163.569206));
	// joints_2d.push_back(cv::Point(174.231231, 213.079903));
	// joints_2d.push_back(cv::Point(183.740234, 278.808601));
	// joints_2d.push_back(cv::Point(233.289581, 158.803978));
	// joints_2d.push_back(cv::Point(242.771667, 225.459358));
	// joints_2d.push_back(cv::Point(281.831085, 277.826485));
	// joints_2d.push_back(cv::Point(205.658874, 122.621582));

	///////////////////////////////////////////////////////////////////////////////////////////
	///////////////////DISPLAY HEATMAP/////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////

	// for(int i = 0; i < hm_avg_input_size.size(); i++)
	// {
	// 	std::string name = "./outputs/vnect/frame2238/heatmap_" + std::to_string(i) + ".jpg";
	// 	cv::imwrite(name, hm_avg_input_size.at(i).mul(1000));
	// }


	// std::ofstream out_oic("outputs/vnect/frame2238/joints_2d.txt"); //img[0][0][0]: -1278951244
	// std::streambuf *coutbuf_oic = std::cout.rdbuf();
	// std::cout.rdbuf(out_oic.rdbuf());
	// 	for(int i = 0; i < joints_2d.size(); i++)
	// 	{
	// 		std::cout << joints_2d.at(i).x << " " << joints_2d.at(i).y << "\n";
	// 	}
	// std::cout.rdbuf(coutbuf_oic);

	// cv::imshow("heatmap_2", hm_avg_input_size.at(2));
	// cv::imshow("heatmap_2x50", hm_avg_input_size.at(2).mul(50));

	///////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////////////

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
	// int limb_parents[21] = {1, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 14, 14, 1, 4, 7, 10, 13};
	
	// for(int i = 0; i < sizeof(limb_parents)/sizeof(limb_parents[0]); i++)
	// {
	// 	int x1 = joints_2d.at(i).x;
	// 	int y1 = joints_2d.at(i).y;
	// 	int x2 = joints_2d.at(limb_parents[i]).x;
	// 	int y2 = joints_2d.at(limb_parents[i]).y;

	// 	float length = pow(pow((x1 - x2),2) + pow((y1 - y2),2), 0.5);
	// 	float deg = atan2(x1-x2, y1-y2);
	// 	deg = (deg*180)/M_PI;
	// }


	//---------------------------------SAVING 3D POINTS TO TEXT FILE---------------------------------//

								/*---- PRINT VNECT_3D_JOINTS.TXT ----*/
					std::ofstream out3djoints("/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/3d_joints/kim_yuna_frame2238.txt");
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
	cv::imshow("image_2d", image_2d);
	cv::imshow("img", img);
	// cv::imwrite("./outputs/vnect/frame2238/image_2d.jpg", image_2d);
	cv::waitKey(0);

	std::cout << "----------done----------\n";
	return 0;
}