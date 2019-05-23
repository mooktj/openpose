#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <openpose/core/common.hpp>
#include <openpose/vnect/vNect.hpp>
#include <openpose/vnect/draw3DPython.hpp>
#include <openpose/vnect/snowMan.hpp>

#include <iostream>

namespace op
{
	cv::Point getFloorLevelPt(std::vector<float> snowMan)
	{	
		int cen_x = snowMan.at(0);
		float top_of_leg = snowMan.at(5);
		float bot_of_leg = snowMan.at(6);
		float rad_bot = bot_of_leg - (top_of_leg + bot_of_leg) / 2;
		cv::Point left_bot(cen_x - rad_bot, (int)bot_of_leg);
		return left_bot;
	}

	float inferDepth(float xlbot_a, float ylbot_a, float xlbot_b, float ylbot_b, cv::Point vanishPt)
	{
		// FIND LENGTH FROM VANISHING POINT TO LEFT_BOT
        float length_a = pow(pow((vanishPt.x - xlbot_a),2) + pow((vanishPt.y - ylbot_a),2), 0.5);
        // float length_b = pow(pow((vanishPt.x - xlbot_b),2) + pow((vanishPt.y - ylbot_b),2), 0.5);

        // cv::circle(currPose_output, vanishPt, 5, cv::Scalar(255,255,0), -1 ,8);
        // cv::circle(currPose_output, cv::Point((int) xlbot_b, (int) ylbot_b), 5, cv::Scalar(0,255,0), -1 ,8);

        float x0 = vanishPt.x; // x of image center
        float y0 = vanishPt.y; // y of image center
        float x1 = xlbot_a; // x of larger snowman left bot
        float y1 = ylbot_a; // y of larger snowman left bot
        float y = ylbot_b; // y of src
        float x = (((y-y0)*(x1-x0))/(y1-y0)) + x0;

        cv::Point start_interpolate((int) xlbot_b, (int) ylbot_b);
        cv::Point end_interpolate((int) x, (int) ylbot_b);
        // cv::line(currPose_output, start_interpolate, end_interpolate, cv::Scalar(0,0,0), 1, 8, 0);
        // cv::circle(currPose_output, end_interpolate, 3, cv::Scalar(0,0,0), -1, 8);

        // FIND INTERPOLATED LENGTH
        float length_x = pow(pow((vanishPt.x - x),2) + pow((vanishPt.y - y),2), 0.5);
        // std::cout << "inferDepth:: length_x: " << length_x << "\n";

		// NORMALISE LENGTH AND GET EXPONENT SCALE
		float norm_factor = 1/length_a;
		// float norm_scale = 5;
    float norm_scale = 2;
		// float norm_length = abs(length_a - length_x) * norm_factor * norm_scale;
    float norm_length = length_x * norm_factor * norm_scale;
    // std::cout << "length_a: " << length_a << ", length_x: " << length_x << "\n";
    // std::cout << "norm_length: " << norm_length << "\n";
		float exp = 2.71828;
		// float infer_depth = pow((10*exp),-(5 - norm_length));
    float infer_depth = pow(exp, (norm_length - 3));

    // if(length_a > length_x) infer_depth =
		return length_a > length_x ? infer_depth : (length_a == length_x ? 0 : -(infer_depth));
	}

  float heightInit(std::vector<float> snowman, cv::Point vanishPt, cv::Point groundPlane)
  {

    // std::cout << "---------HEIGHT INIT---------\n";
    cv::Point top((int) snowman.at(0), (int) snowman.at(1));
    cv::Point bot((int) snowman.at(0), (int) snowman.at(6));

    float x0 = vanishPt.x; // x of image center
    float y0 = vanishPt.y; // y of image center
    float xtop = snowman.at(0);
    float ytop = snowman.at(1);
    float xbot = snowman.at(0);
    float ybot = snowman.at(6);
    float yinbot = groundPlane.y;
    float xinbot = (((ybot - y0)*(xbot - x0))/(ybot - y0)) + x0;

    float xintop = xinbot;
    float yintop = (((xintop - x0)*(ytop - y0))/(xtop - x0)) + y0;

    // std::cout << "xtop, ytop: " << xtop << "," << ytop << "\n";
    // std::cout << "xbot, ybot: " << xbot << "," << ybot << "\n";
    // std::cout << "xintop, yintop: " << xintop << "," << yintop << "\n";
    // std::cout << "xinbot, yinbot: " << xinbot << "," << yinbot << "\n";

    cv::Point topEx((int) xintop, (int) yintop);
    cv::Point botEx((int) xinbot, (int) yinbot);

    std::vector<cv::Point> ret;

    ret.push_back(top); // 0
    ret.push_back(bot); // 1
    ret.push_back(topEx); // 2
    ret.push_back(botEx); // 3

    // std::cout << "top: (" << top.x << "," << top.y << ")\n";
    // std::cout << "bot: (" << bot.x << "," << bot.y << ")\n";
    // std::cout << "topEx: (" << topEx.x << "," << topEx.y << ")\n";
    // std::cout << "botEx: (" << botEx.x << "," << botEx.y << ")\n";
    std::cout << "height of extrapolated: " << botEx.y - topEx.y << "\n";

    return botEx.y - topEx.y;
  }

	float compareSnowmenSize(std::vector<float> currSnow, std::vector<float> prevSnow)
	{
		// ------ CURR SNOW PARAMETERS ------ //
		float cen_x = currSnow.at(0);
        float top_of_head = currSnow.at(1);
        float bot_of_head = currSnow.at(2);
        float top_of_mid = currSnow.at(3);
        float bot_of_mid = currSnow.at(4);
        float top_of_leg = currSnow.at(5);
        float bot_of_leg = currSnow.at(6);

        float rad_top =  bot_of_head - (top_of_head + bot_of_head) / 2;
        float rad_mid =  bot_of_mid - (top_of_mid + bot_of_mid) / 2;
        float rad_bot = bot_of_leg - (top_of_leg + bot_of_leg) / 2;

        // ------ PREV SNOW PARAMETERS ------ //
        float cen_x_p = prevSnow.at(0);
        float top_of_head_p = prevSnow.at(1);
        float bot_of_head_p = prevSnow.at(2);
        float top_of_mid_p = prevSnow.at(3);
        float bot_of_mid_p = prevSnow.at(4);
        float top_of_leg_p = prevSnow.at(5);
        float bot_of_leg_p = prevSnow.at(6);

        float rad_top_p =  bot_of_head_p - (top_of_head_p + bot_of_head_p) / 2;
        float rad_mid_p =  bot_of_mid_p - (top_of_mid_p + bot_of_mid_p) / 2;
        float rad_bot_p = bot_of_leg_p - (top_of_leg_p + bot_of_leg_p) / 2;

        float diff_cen = cen_x - cen_x_p;
        float diff_top = rad_top - rad_top_p;
        float diff_mid = rad_mid - rad_mid_p;
        float diff_bot = rad_bot - rad_bot_p;

        // if(diff_top == 0 && diff_mid == 0 && diff_bot == 0)
        // {
        //     std::cout << "----------------got exact same pose-----------------\n";
        // }

        float score = (3 * diff_cen) + (3 * diff_mid) + (2 * diff_top) + (1 * diff_bot);
        // std::cout << "->score: " << score << "\n";
        return score;
	}

	float reMatchSnowmen(std::vector<float> prev, std::vector<float> curr)
	{
		int cen_x_p = prev.at(0);
        float top_of_head_p = prev.at(1);
        float bot_of_head_p = prev.at(2);
        float top_of_mid_p = prev.at(3);
        float bot_of_mid_p = prev.at(4);
        float top_of_leg_p = prev.at(5);
        float bot_of_leg_p = prev.at(6);


            float x_head_p = prev.at(7);
            float y_head_p = prev.at(8);

            float x_neck_p = prev.at(9);
            float y_neck_p = prev.at(10);

            float x_chest_p = prev.at(11);
            float y_chest_p = prev.at(12);

            float x_lhip_p = prev.at(13);
            float y_lhip_p = prev.at(14);

            float x_rhip_p = prev.at(15);
            float y_rhip_p = prev.at(16);

            float x_lknee_p = prev.at(17);
            float y_lknee_p = prev.at(18);

            float x_rknee_p = prev.at(19);
            float y_rknee_p = prev.at(20);

            float x_lankle_p = prev.at(21);
            float y_lankle_p = prev.at(22);

            float x_rankle_p = prev.at(23);
            float y_rankle_p = prev.at(24);


        int cen_x_c = curr.at(0);
        float top_of_head_c = curr.at(1);
        float bot_of_head_c = curr.at(2);
        float top_of_mid_c = curr.at(3);
        float bot_of_mid_c = curr.at(4);
        float top_of_leg_c = curr.at(5);
        float bot_of_leg_c = curr.at(6);

            float x_head_c = curr.at(7);
            float y_head_c = curr.at(8);

            float x_neck_c = curr.at(9);
            float y_neck_c = curr.at(10);

            float x_chest_c = curr.at(11);
            float y_chest_c = curr.at(12);

            float x_lhip_c = curr.at(13);
            float y_lhip_c = curr.at(14);

            float x_rhip_c = curr.at(15);
            float y_rhip_c = curr.at(16);

            float x_lknee_c = curr.at(17);
            float y_lknee_c = curr.at(18);

            float x_rknee_c = curr.at(19);
            float y_rknee_c = curr.at(20);

            float x_lankle_c = curr.at(21);
            float y_lankle_c = curr.at(22);

            float x_rankle_c = curr.at(23);
            float y_rankle_c = curr.at(24);


		// 1. SEE WHICH POSEKEYPOINTS ARE DETECTED FROM THE Y HEIGHT AND RADIUS OF THE SNOWMAN

            float threshold = 20; // 20 pixels
      			// CHECK THE LOCATIONS WHETHER WITHIN THRESHOLD
            // BOT
      		float xlankle_d = (x_lankle_c == 0) ? 100 : (x_lankle_p - x_lankle_c); // in threshold?
      			if((x_lankle_c != 0) && (xlankle_d > threshold)) return -1;
      		float ylankle_d = (y_lankle_c == 0) ? 100 : (y_lankle_p - y_lankle_c); // in threshold?
      			if((y_lankle_c != 0) && (ylankle_d > threshold)) return -1;
      		float xrankle_d = (x_rankle_c == 0) ? 100 : (x_rankle_p - x_rankle_c); // in threshold?
      			if((x_rankle_c != 0) && (xrankle_d > threshold)) return -1;
      		float yrankle_d = (y_rankle_c == 0) ? 100 : (y_rankle_p - y_rankle_c); // in threshold?
      			if((y_rankle_c != 0) && (yrankle_d > threshold)) return -1;

      		float xlknee_d = (x_lknee_c == 0) ? 100 : (x_lknee_p - x_lknee_c);
      			if((x_lknee_c != 0) && (xlknee_d > threshold)) return -1;
      		float ylknee_d = (y_lknee_c == 0) ? 100 : (y_lknee_p - y_lknee_c);
      			if((y_lknee_c != 0) && (ylknee_d > threshold)) return -1;
      		float xrknee_d = (x_rknee_c == 0) ? 100 : (x_rknee_p - x_rknee_c);
      			if((x_rknee_c != 0) && (xrknee_d > threshold)) return -1;
      		float yrknee_d = (y_rknee_c == 0) ? 100 : (y_rknee_p - y_rknee_c);
      			if((y_rknee_c != 0) && (yrknee_d > threshold)) return -1;
      		
      		float xlhip_d = (x_lhip_c == 0) ? 100 : (x_lhip_p - x_lhip_c);
      			if((x_lhip_c != 0) && (xlhip_d > threshold)) return -1;
      		float ylhip_d = (y_lhip_c == 0) ? 100 : (y_lhip_p - y_lhip_c);
      			if((y_lhip_c != 0) && (ylhip_d > threshold)) return -1;
      		float xrhip_d = (x_rhip_c == 0) ? 100 : (x_rhip_p - x_rhip_c);
      			if((x_rhip_c != 0) && (xrhip_d > threshold)) return -1;
      		float yrhip_d = (y_rhip_c == 0) ? 100 : (y_rhip_p - y_rhip_c);
      			if((y_rhip_c != 0) && (yrhip_d > threshold)) return -1;

      		float botleg_d = (bot_of_leg_c == 0) ? 100 : (bot_of_leg_p - bot_of_leg_c); // in threshold?
      			if((bot_of_leg_c != 0) && (botleg_d > threshold)) return -1;
      		float topleg_d = (top_of_leg_c == bot_of_leg_c) ? 100 : (top_of_leg_p - bot_of_leg_c); // in threshold?
      			if((top_of_leg_c != bot_of_leg_c) && (topleg_d > threshold)) return -1;

      		// MID
      		float xchest_d = (x_chest_c == 0) ? 100 : (x_chest_p - x_chest_c);
      			if((x_chest_c != 0) && (xchest_d > threshold)) return -1;
      		float ychest_d = (y_chest_c == 0) ? 100 : (y_chest_p - y_chest_c);
      			if((y_chest_c != 0) && (ychest_d > threshold)) return -1;

      		float botmid_d = (bot_of_mid_c == 0) ? 100 : (bot_of_mid_p - bot_of_mid_c);
      			if((bot_of_mid_c != 0) && (botmid_d > threshold)) return -1;
      		float topmid_d = (top_of_mid_c == bot_of_mid_c) ? 100 : (top_of_mid_p - top_of_mid_c);
      			if((top_of_mid_c != bot_of_mid_c) && (botmid_d > threshold)) return -1;

      		// TOP
      		float xneck_d = (x_neck_c == 0) ? 100 : (x_neck_p - x_neck_c);
      			if((x_neck_c != 0) && (xneck_d > threshold)) return -1;
      		float yneck_d = (y_neck_c == 0) ? 100 : (y_neck_p - y_neck_c);
      			if((y_neck_c != 0) && (yneck_d > threshold)) return -1;

      		float xhead_d = (x_head_c == 0) ? 100 : (x_head_p - x_head_c);
      			if((x_head_c != 0) && (xhead_d > threshold)) return -1;
      		float yhead_d = (y_head_c == 0) ? 100 : (y_head_p - y_head_c);
      			if((y_head_c != 0) && (yhead_d > threshold)) return -1;

      		float bothead_d = (bot_of_head_c == 0) ? 100 : (bot_of_head_p - bot_of_head_c);
      			if((bot_of_head_c != 0) && (bothead_d > threshold)) return -1;
      		float tophead_d = (top_of_head_c == bot_of_head_c) ? 100 : (top_of_head_p - top_of_head_c);
      			if((top_of_head_c != bot_of_head_c) && (tophead_d > threshold)) return -1;

      		// CHECK WHETHER ANY VALID DIFF GOES BEYOND THRESHOLD

			float bot_score = abs(xlankle_d) + abs(ylankle_d) + abs(xrankle_d) + abs(yrankle_d) + 
						abs(xlknee_d) + abs(ylknee_d) + abs(xrknee_d) + abs(yrknee_d) + 
						abs(xlhip_d) + abs(ylhip_d) + abs(xrhip_d) + abs(yrhip_d) + 
						abs(botleg_d) + abs(topleg_d);

			float mid_score = abs(xlhip_d) + abs(ylhip_d) + abs(xrhip_d) + abs(yrhip_d) + 
						abs(xchest_d) + abs(ychest_d) +
						abs(botmid_d) + abs(topmid_d);

			float top_score = abs(xchest_d) + abs(ychest_d) +
						abs(xneck_d) + abs(yneck_d) +
						abs(xhead_d) + abs(yhead_d) +
						abs(bothead_d) + abs(tophead_d);

			float score = ((1) * bot_score) + ((1.5) * mid_score) + ((1.75) * top_score);

		// 2. IF THE LOCATIONS ARE BELOW THE THRESHOLD, STORE THE INDEX AND SCORE

		return score;
	}

	std::vector<float> updateSnowman(std::vector<float> prevSnowman, std::vector<float> currSnowman, cv::Point groundPlane)
	{
		std::vector<float> out;

		// std::cout << "size of currin vector: " << currSnowman.size() << "\n";
    // std::cout << "----------------------->      prev size: " << prevSnowman.size() << "\n";
		for(auto c = 0; c < currSnowman.size(); c++)
		{
      // std::cout << "1. c: " << c << "\n";
			if(c >= 7 && c <= 24)
			{
				if(c % 2 != 1) continue;
          // std::cout << "2. c: " << c << "\n";
				bool newLoc = (currSnowman.at(c) == 0) || (currSnowman.at(c+1) == 0);
				// std::cout << "newLoc: " << newLoc << "\n";
				float xLoc = newLoc ? prevSnowman.at(c) : currSnowman.at(c);// (prevSnowman.at(c) + currSnowman.at(c))/2;
				float yLoc = newLoc ? prevSnowman.at(c+1) : currSnowman.at(c+1); // (prevSnowman.at(c+1) + currSnowman.at(c+1))/2;
				out.push_back(xLoc);
				out.push_back(yLoc);
			}
			else 
			{
				out.push_back(currSnowman.at(c));
			}
		}
    // std::cout << "updateSnowman:: prevSnowman(28): " << prevSnowman.at(28) << "\n";
    
    // std::cout << "updateSnowman::------OUT OF COPYING CURRSNOWMAN\n";
    // for(auto c = 0; c < currSnowman.size(); c++)
    // {
    //   out.push_back(currSnowman.at(c));
    // }

		// GET DEPTH BY COMPARING THE RADIUS OF PREV SNOWMAN
		    int cen_x_p = prevSnowman.at(0);
        // std::cout << "cen_x_p\n";
        float top_of_head_p = prevSnowman.at(1);
        // std::cout << "top_of_head_p\n";
        float bot_of_head_p = prevSnowman.at(2);
        // std::cout << "bot_of_head_p\n";
        float top_of_mid_p = prevSnowman.at(3);
        // std::cout << "top_of_mid_p\n";
        float bot_of_mid_p = prevSnowman.at(4);
        // std::cout << "bot_of_mid_p\n";
        float top_of_leg_p = prevSnowman.at(5);
        // std::cout << "top_of_leg_p\n";
        float bot_of_leg_p = prevSnowman.at(6);
        // std::cout << "bot_of_leg_p\n";

        int cen_x_c = currSnowman.at(0);
        // std::cout << "cen_x_c\n";
        float top_of_head_c = currSnowman.at(1);
        // std::cout << "top_of_head_c\n";
        float bot_of_head_c = currSnowman.at(2);
        float top_of_mid_c = currSnowman.at(3);
        float bot_of_mid_c = currSnowman.at(4);
        float top_of_leg_c = currSnowman.at(5);
        float bot_of_leg_c = currSnowman.at(6);
        // std::cout << "bot_of_leg_c\n";

		    float rad_top_p =  bot_of_head_p - (top_of_head_p + bot_of_head_p) / 2;
        float rad_mid_p =  bot_of_mid_p - (top_of_mid_p + bot_of_mid_p) / 2;
        float rad_bot_p = bot_of_leg_p - (top_of_leg_p + bot_of_leg_p) / 2;

		    float rad_top_c = bot_of_head_c - (top_of_head_c + bot_of_head_c) / 2;
        float rad_mid_c = bot_of_mid_c - (top_of_mid_c + bot_of_mid_c) / 2;
        float rad_bot_c = bot_of_leg_c - (top_of_leg_c + bot_of_leg_c) / 2;

        float rad_top = (bot_of_head_c == top_of_head_c) ? rad_top_p : rad_top_c;
        float rad_mid = (bot_of_mid_c == top_of_mid_c) ? rad_mid_p : rad_mid_c;
        float rad_bot = (bot_of_leg_c == top_of_leg_c) ? rad_bot_p : rad_bot_c;

		if(cen_x_c == 0) 
		{
			// std::cout << "..cen_x_c = 0\n";
			out.at(0) = prevSnowman.at(0); // set cen_x
		}

		if(bot_of_leg_c == 0) 
		{
			// std::cout << "..bot_of_leg_c = 0\n";
			out.at(6) = prevSnowman.at(6); // set bot_of_leg_c
      out.at(5) = out.at(6) - (2 * rad_bot); // set top_of_leg_c
      out.at(4) = out.at(5); // set bot_of_mid_c
      out.at(3) = out.at(4) - (2 * rad_mid); // set top_of_mid_c
      out.at(2) = out.at(3); // set bot_of_head_c
      out.at(1) = out.at(2) - (2 * rad_top); // set top_of_head_c
		}

		if(bot_of_leg_c == top_of_leg_c) 
		{
			// std::cout << "..bot_of_leg_c == top_of_leg_c\n";
			out.at(5) = out.at(6) - (2 * rad_bot); // set top_of_leg_c
			out.at(4) = out.at(5); // set bot_of_mid_c
			out.at(3) = out.at(4) - (2 * rad_mid); // set top_of_mid_c
			out.at(2) = out.at(3); // set bot_of_head_c
			out.at(1) = out.at(2) - (2 * rad_top); // set top_of_head_c
		}
		if(bot_of_mid_c == top_of_mid_c)
		{
			// std::cout << "..bot_of_mid_c == top_of_mid_c\n";
			out.at(3) = out.at(4) - (2 * rad_mid); // set top_of_mid_c
			out.at(2) = out.at(3); // set bot_of_head_c
			out.at(1) = out.at(2) - (2 * rad_top); // set top_of_head_c
		}
		if(bot_of_head_c == top_of_head_c)
		{
			// std::cout << "..bot_of_head_c == top_of_head_c\n";
			out.at(1) = out.at(2) - (2 * rad_top); // set top_of_head_c
		}


    // ----------- ADJUST DEPTH ----------- //
            // Denominator must never be 0 since prev snowman must be a perfect snowman
        // std::cout << "bot_diff --\n";
        // std::cout << "abs(bot_of_leg_c - top_of_leg_c): " << abs(bot_of_leg_c - top_of_leg_c) << "\n"; 
        // std::cout << "abs(bot_of_leg_p - top_of_leg_p): " << abs(bot_of_leg_p - top_of_leg_p) << "\n";
        // float bot_diff = (abs(bot_of_leg_c - top_of_leg_c) / abs(bot_of_leg_p - top_of_leg_p));
        float bot_diff = ((bot_of_leg_c - top_of_leg_c) / (bot_of_leg_p - top_of_leg_p));

        // std::cout << "mid_diff\n";
        // std::cout << "abs(bot_of_mid_c - top_of_mid_c): " << abs(bot_of_mid_c - top_of_mid_c) << "\n";
        // std::cout << "abs(bot_of_mid_p - top_of_mid_p): " << abs(bot_of_mid_p - top_of_mid_p) << "\n";
        // float mid_diff = (abs(bot_of_mid_c - top_of_mid_c) / abs(bot_of_mid_p - top_of_mid_p));
        float mid_diff = ((bot_of_mid_c - top_of_mid_c) / (bot_of_mid_p - top_of_mid_p));

        // std::cout << "top_diff\n";
        // std::cout << "abs(bot_of_head_c - top_of_head_c): " << abs(bot_of_head_c - top_of_head_c) << "\n";
        // std::cout << "abs(bot_of_head_p - top_of_head_p): " << abs(bot_of_head_p - top_of_head_p) << "\n";
        // float top_diff = (abs(bot_of_head_c - top_of_head_c) / abs(bot_of_head_p - top_of_head_p));
        float top_diff = ((bot_of_head_c - top_of_head_c) / (bot_of_head_p - top_of_head_p));

        float sizeWeight = (0.1 * bot_diff) + (mid_diff) + (0.2 * top_diff);
        // std::cout << "depthAdjust: " << depthAdjust << "\n";
        // std::cout << "prevSnowman.at(27): " << prevSnowman.at(27) << "\n";
        // std::cout << "new depth: " << prevSnowman.at(27) * depthAdjust << "\n";

        float floorW = 0; float exp = 2.71828;
        // IF GOING INTO THE IMAGE (I.E. HIGHER GROUND HEIGHT)
          // if((int)out.at(6) < (int)prevSnowman.at(6)) // (5 -> 4, 3, 2)
          // {
          //   // 2*(e^x) where x = out[6]/prevSnowman[6] - 2
          //   float x = x = (out.at(6)/prevSnowman.at(6)) - 2;
          //   floorW = -(2 * pow(exp, x));
          //   std::cout << "out < prev" << "\n";
          //   std::cout << "x: " << x << "\n"; 
          // }
          // // IF COMING OUT OF THE IMAGE (I.E. LOWER GROUND HEIGHT)
          // else if((int)out.at(6) > (int)prevSnowman.at(6))
          // {
          //   float x = out.at(6) - prevSnowman.at(6);
          //   floorW = pow(exp, x);
          //   std::cout << "out > prev" << "\n";
          //   std::cout << "x: " << x << "\n";
          // }
        // IF STAY THE SAME (I.E. SAME GROUND HEIGHT)
        // else if(out.at(6) == prevSnowman.at(6))
        // {
        //   floorW = 0;
        // }
        // std::cout << "out.at(6): " << out.at(6) << ", prevSnowman.at(6): " << prevSnowman.at(6) << "\n";
        // std::cout << " diff: " << prevSnowman.at(6) - out.at(6) << "\n";
        // std::cout << "floorW: " << floorW << "\n";
        // float floorHeight = prevSnowman.at(27) + (prevSnowman.at(27) * floorW);
        float floorHeight = prevSnowman.at(27) + (prevSnowman.at(6) - out.at(6));
        float depthAdjust = (1 * floorHeight) + (0 * sizeWeight);
        out.push_back(depthAdjust);
        // out.push_back(prevSnowman.at(27) * 1);

		// std::cout << "size of out vector: " << out.size() << "\n";
  //       std::cout << "prev depth: " << prevSnowman.at(27) << ", depthAdjust: " << depthAdjust << "\n";
        out.push_back(prevSnowman.at(28));
        // std::cout << "out at (28): " << out.at(28) << "\n";
		return out;
	}

	std::vector<float> reAdjustSnowman(std::vector<float> prevSnowman, std::vector<float> currSnowman)
	{
		// std::vector<float> out;

		return prevSnowman;
	}

	bool checkFullSnowman(std::vector<float> currSnowman)
	{
		// std::vector<float> out;
				// CHECK IF THE MATCH IS APPROPRIATE, NOT JUST NUMERICALLY MATCHED
        float top_of_head_c = currSnowman.at(1);
        float bot_of_head_c = currSnowman.at(2);
			if(top_of_head_c == bot_of_head_c) return false;
        float top_of_mid_c = currSnowman.at(3);
        float bot_of_mid_c = currSnowman.at(4);
			if(top_of_mid_c == bot_of_mid_c) return false;
        float top_of_leg_c = currSnowman.at(5);
        float bot_of_leg_c = currSnowman.at(6);
			if(top_of_leg_c == bot_of_leg_c) return false;
   //      int cen_x_c = currSnowman.at(0);
			// if(cen_x_c == 0) return;
      if(bot_of_leg_c <= 0) return false;
		return true;
	}


}