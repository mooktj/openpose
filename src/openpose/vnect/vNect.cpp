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
        for(auto i = 0; i < in.size(); i++)
        {
            if(max < in.at(i)) max = in.at(i);
        }
        return max;
    }

    int getMaxIndex(std::vector<float> in, float max_value)
    {
        // float max = 0;
        auto i = 0;
        for(i = 0; i < in.size(); i++)
        {
            if(max_value = in.at(i)) break;
        }
        return (int) i;
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
        // std::cout << "________vNectPostForward________\n";
    	auto& poseKeypoints = datumsPtr->poseKeypoints;
    	std::vector<float> poseDistances;

		// std::vector<std::vector<float>> jointsRad;
		// std::vector<std::vector<cv::Point>> jointsCen;
        std::vector<std::vector<float>> snowmen;

		cv::Mat currPose = datumsPtr->cvOutputData.clone();
		cv::Mat currPose_output = datumsPtr->cvOutputData.clone();
		float opacity = 0.5;

		for(int i = 0; i < poseKeypoints.getSize(0); i++)
        {

            // std::vector<float> currRad;
            // std::vector<cv::Point> currCen;
            std::vector<float> snowman;

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////----TOP----////////////////////////////////////////////////////////////////
            float x_neck = poseKeypoints[{i,1,0}];
            float y_neck = poseKeypoints[{i,1,1}];

            float x_head = poseKeypoints[{i,0,0}];
            float y_head = poseKeypoints[{i,0,1}];

            float x_chest = poseKeypoints[{i,14,0}];
            float y_chest = poseKeypoints[{i,14,1}];

            float l_head = pow(pow((x_head - x_neck),2) + pow((y_head - y_neck),2), 0.5);
            float l_chest = pow(pow((x_chest - x_neck),2) + pow((y_chest - y_neck),2), 0.5);

            // choose larger radius between head and chest
            float rad_top = l_head > l_chest ? l_head : l_chest;
            // currRad.push_back(rad_top);

            cv::Point cen_top = {(int)x_neck,(int)y_neck};
            // currCen.push_back(cen_top);

            // cv::circle(currPose, cen_top, rad_top, cv::Scalar(0,0,255), -1, 8);
            // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);

            //------------GET TOP and BOT OF HEAD-------------//
            // float top_of_head = y_neck - rad_top;
            // float bot_of_head = y_neck + rad_top;
            // cv::circle(currPose, cen_top, rad_top, cv::Scalar(0,0,255), -1, 8);
            // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);
            // cv::circle(currPose, cv::Point((int)x_neck, (int)top_of_head), 5, cv::Scalar(200,0,190), -1, 8);
            // cv::circle(currPose, cv::Point((int)x_neck, (int)bot_of_head), 5, cv::Scalar(0,200,190), -1, 8);

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////////

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////----MID----////////////////////////////////////////////////////////////////
            float x_lhip = poseKeypoints[{i,11,0}];
            float y_lhip = poseKeypoints[{i,11,1}];

            float x_rhip = poseKeypoints[{i,8,0}];
            float y_rhip = poseKeypoints[{i,8,1}];

            std::cout << "--* x_rhip: " << x_rhip << "\n";
            std::cout << "--* y_rhip: " << y_rhip << "\n";

            float dia_lmid = pow(pow((x_chest - x_lhip),2) + pow((y_chest - y_lhip),2), 0.5); // diameter of chest/lhip
            if(x_chest == 0 || x_lhip == 0 || y_chest == 0 || y_lhip == 0)
            {
                dia_lmid = 0;
            }
            float rad_lmid = dia_lmid/2;
            // currRad.push_back(rad_lmid);

            cv::Point cen_lmid = {(int)(x_chest + x_lhip)/2, (int)(y_chest + y_lhip)/2};
            // currCen.push_back(cen_lmid);

            // if(x_chest != 0 && y_chest != 0 && x_lhip != 0 && y_lhip != 0)
            // {
            //     cv::circle(currPose, cen_lmid, rad_lmid, cv::Scalar(255,0,0), -1, 8);
            //     cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            // }

            float dia_rmid = pow(pow((x_chest - x_rhip),2) + pow((y_chest - y_rhip),2), 0.5); // diameter of chest/rhip
            float rad_rmid = dia_rmid/2;
            // currRad.push_back(rad_rmid);

            cv::Point cen_rmid = {(int)(x_chest + x_rhip)/2, (int)(y_chest + y_rhip)/2};
            // currCen.push_back(cen_rmid);

            // if(x_chest != 0 && y_chest != 0 && x_rhip != 0 && y_rhip != 0)
            // {
            //     cv::circle(currPose, cen_rmid, rad_rmid, cv::Scalar(255,0,0), -1, 8);
            //     cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            // }             

            float dia_mid;
            if(dia_lmid == 0 && dia_rmid == 0) dia_mid = 0;
            else if(dia_lmid == 0 && dia_rmid != 0) dia_mid = dia_rmid;
            else if(dia_rmid == 0 && dia_lmid != 0) dia_mid = dia_lmid;
            else dia_mid = dia_lmid > dia_rmid ? dia_lmid : dia_rmid;

            //------------GET TOP and BOT OF MID-------------//
            // float top_of_mid = bot_of_head;
            // float bot_of_mid = top_of_mid + dia_mid;

            // cv::circle(currPose, cv::Point((int)x_neck, (int)top_of_mid), 5, cv::Scalar(0,20,190), -1, 8);
            // cv::circle(currPose, cv::Point((int)x_neck, (int)bot_of_mid), 5, cv::Scalar(0,20,190), -1, 8);
            //* TODO: HANDLE WHEN dia_mid = 0, get dia_mid from previous or original dia_mids

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////////

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ////////////////////////////----BOT----////////////////////////////////////////////////////////////////            
            float x_lankle = poseKeypoints[{i,13,0}];
            float y_lankle = poseKeypoints[{i,13,1}];
            std::cout << "..^ x_lankle: " << x_lankle << "\n";
            std::cout << ".. y_lankle: " << y_lankle << "\n";

            float x_rankle = poseKeypoints[{i,10,0}];
            float y_rankle = poseKeypoints[{i,10,1}];

            std::cout << "..^ x_rankle: " << x_rankle << "\n";
            std::cout << ".. y_rankle: " << y_rankle << "\n";

            float x_lknee = poseKeypoints[{i,12,0}];
            float y_lknee = poseKeypoints[{i,12,1}];

            float x_rknee = poseKeypoints[{i,9,0}];
            float y_rknee = poseKeypoints[{i,9,1}];

            float l_lhip = pow(pow((x_lhip - x_lknee),2) + pow((y_lhip - y_lknee),2), 0.5);
            float l_lankle = pow(pow((x_lankle - x_lknee),2) + pow((y_lankle - y_lknee),2), 0.5);

            float l_rhip = pow(pow((x_rhip - x_rknee),2) + pow((y_rhip - y_rknee),2), 0.5);
            float l_rankle = pow(pow((x_rankle - x_rknee),2) + pow((y_rankle - y_rknee),2), 0.5);

            // choosing larger radius for LEFT BOT
            float rad_lbot = l_lhip > l_lankle ? l_lhip : l_lankle;
            // currRad.push_back(rad_lbot);

            cv::Point cen_lbot = {(int)x_lknee, (int)y_lknee};
            // currCen.push_back(cen_lbot);

            if(x_lknee != 0 && y_lknee != 0)
            {
                // cv::circle(currPose, cen_lbot, rad_lbot, cv::Scalar(0,255,0), -1, 8);
                    // cv::line(currPose, cv::Point(center.x - 25, center.y+radius), cv::Point(center.x + 25, center.y+radius), cv::Scalar(0,255,0), 1, 8, 0);
                    // std::string text = "h: " + std::to_string(center.y + radius);
                    // cv::Point textOrg = cv::Point(center.x + 27, center.y + radius - 5);
                    // int fontFace = CV_FONT_HERSHEY_SIMPLEX;
                    // cv::putText(currPose, text, textOrg, fontFace, 0.3, cv::Scalar(0,0,0), 1, 8);
                // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);
            }

            // choosing larger radius for RIGHT BOT
            float rad_rbot = l_rhip > l_rankle ? l_rhip : l_rankle;
            // currRad.push_back(rad_rbot);
            std::cout << "      --l_rhip: " << l_rhip << "\n";
            std::cout << "l_rankle: " << l_rankle << "\n";
            std::cout << "rad_rbot: " << rad_rbot << "\n";

            cv::Point cen_rbot = {(int)x_rknee, (int)y_rknee};
            // currCen.push_back(cen_rbot);

            // std::cout << "x_rknee: " << x_rknee << "\n";
            // std::cout << "y_rknee: " << y_rknee << "\n";
            // if(x_rknee != 0 && y_rknee != 0)
            // {
                // cv::circle(currPose, cen_rbot, rad_rbot, cv::Scalar(255,0,0), -1, 8);
                // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            // }

            // can be different leg from lower_ankle_height because both legs are normally of the same/similar length
            float rad_bot = rad_lbot > rad_rbot ? rad_lbot : rad_rbot;
            float dia_leg = 2 * rad_bot;
            // float top_of_leg = bot_of_mid;
            // float bot_of_leg = top_of_leg + dia_leg;

            std::cout << "      --rad_lbot: " << rad_lbot << "\n";
            std::cout << "rad_rbot: " << rad_rbot << "\n";
            std::cout << "rad_bot: " << rad_bot << "\n";

            ////////////////////*TO-DO: CONSIDER AVERAGE FLOOR LEVEL OF TWO ANKLES IF WITHIN 5% BOUND///////////////////////////
            // float higher_ankle_height = y_lankle < y_rankle ? y_lankle : y_rankle;

            // // find 5% tolerance bound within lower_ankle_height
            // float five_percent = (lower_ankle_height * 0.05);
            // float upper_bound = abs(lower_ankle_height - five_percent);
            // float lower_bound = abs(lower_ankle_height + five_percent);

            // // if higher_ankle_height is within 5% of lower ankle_height, average the floor level height
            // if(upper_bound < higher_ankle_height < lower_bound)
            // {
            //     int avg_x = (int)(jointsCenExtr.at(i).at(1).x + jointsCenExtr.at(i).at(2).x)/2;
            //     int avg_y = (int)(jointsCenExtr.at(i).at(1).y + jointsCenExtr.at(i).at(2).y)/2;
            //     cv::circle(currPose_output, cv::Point(jointsCenExtr.at(i).at(0).x, avg_y),5,cv::Scalar(180,180,180),-1,8);
            // }
            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            ///////////////////////////////////////////////////////////////////////////////////////////////////////
            ///////////////////////////////////////////////////////////////////////////////////////////////////////

            // std::cout << "(int)(cen_lmid.x + cen_rmid.x)/2 : " << (int)(cen_lmid.x + cen_rmid.x)/2 << "\n";
            //------------GET TOP and BOT OF LEG-------------//
            std::cout << "  ---->GET TOP BOT LEG\n";
            float lower_ankle_height = y_lankle > y_rankle ? y_lankle : y_rankle;
            // int x_leg = (int) (y_lankle > y_rankle ? x_lankle : x_rankle);

            // std::cout << "~~y_lankle: " << y_lankle << "\n";
            // std::cout << "y_rankle: " << y_rankle << "\n";
            // std::cout << "lower_ankle_height: " << lower_ankle_height << "\n";

            float bot_of_leg = lower_ankle_height;
            float top_of_leg = bot_of_leg - dia_leg;

            std::cout << "bot_of_leg: " << bot_of_leg << "\n";
            std::cout << "top_of_leg: " << top_of_leg << "\n";
            std::cout << "dia_leg: " << dia_leg << "\n";

            // if(i == 1)

            //------------GET TOP and BOT OF MID-------------//
            // std::cout << "---->GET TOP BOT MID\n";
            float bot_of_mid = top_of_leg;
            float top_of_mid = bot_of_mid - dia_mid;

            // std::cout << "bot_of_mid: " << bot_of_mid << "\n";
            // std::cout << "top_of_mid: " << top_of_mid << "\n";
            

            //------------GET TOP and BOT OF HEAD-------------//
            // std::cout << "---->GET TOP BOT HEAD\n";
            float bot_of_head = top_of_mid;
            float top_of_head = bot_of_head - (2 * rad_top);

            // std::cout << "bot_of_head: " << bot_of_head << "\n";
            // std::cout << "top_of_head: " << top_of_head << "\n";

            // if(i == 1)
            // {

            // std::cout << "rad_bot: " << rad_bot << "\n";
            // std::cout << "dia_mid/2: " << dia_mid/2 << "\n";
            // std::cout << "rad_top: " << rad_top << "\n";
                int cen_x = (int)(cen_lmid.x + cen_rmid.x)/2;

                // cv::circle(currPose, cv::Point(cen_x, (int) (top_of_leg + bot_of_leg)/2), rad_bot, cv::Scalar(255,255,0), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int) (top_of_mid + bot_of_mid)/2), dia_mid/2, cv::Scalar(255,255,0), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int) (top_of_head + bot_of_head)/2), rad_top, cv::Scalar(255,255,0), -1, 8);
                // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output);

                // // cen_x = 200;
                // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)top_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)top_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_head), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)top_of_head), 5, cv::Scalar(0,255,255), -1, 8);


                // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            // }

            // EXTRACT SNOWMAN
            snowman.push_back(float(cen_x));// 0. cen_x
            snowman.push_back(top_of_head); // 1. top_of_head
            snowman.push_back(bot_of_head); // 2. bot_of_head
            snowman.push_back(top_of_mid);  // 3. top_of_mid
            snowman.push_back(bot_of_mid);  // 4. bot_of_mid
            snowman.push_back(top_of_leg);  // 5. top_of_leg
            snowman.push_back(bot_of_leg);  // 6. bot_of_leg

            // cannot use as rads because may detach snowman
            // snowman.push_back(rad_top);     // 7. rad_top
            // snowman.push_back(dia_mid/2);   // 8. rad_mid
            // snowman.push_back(rad_bot);     // 9. rad_bot

            snowmen.push_back(snowman);


            // jointsRad.push_back(currRad);
            // jointsCen.push_back(currCen);

        }

        float lowest_snowman = -1;
        std::vector<std::vector<float>> snowmen_sorted;
        // std::vector<int>::iterator it;

        // it = snowmen_sorted.begin();

        // REORDER JOINTS_3D POSES
        std::vector<std::vector<std::vector<float>>> joints_3d;
        std::vector<std::vector<std::vector<float>>> datumsPtr_joints_3d = datumsPtr->joints_3d_root_relative;

        // REORDER OPENPOSE 2D POSES
        std::vector<std::vector<std::vector<float>>> joints_2d_orig;
        std::vector<std::vector<std::vector<float>>> joints_2d;


        // CONVERT POSEKEYPOINTS ARRAY TO STD::VECTOR
        for(auto pose = 0; pose < poseKeypoints.getSize(0); pose++)
        {
            std::vector<std::vector<float>> joints;
            for(auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
            {
                // std::cout << "bodyPart: " << bodyPart << "\n";
                std::vector<float> joint;
                joint.push_back(poseKeypoints[{pose, bodyPart, 0}]);
                joint.push_back(poseKeypoints[{pose, bodyPart, 1}]);
                joint.push_back(poseKeypoints[{pose, bodyPart, 2}]);
                joints.push_back(joint);
            }
            joints_2d_orig.push_back(joints);
        }

        // std::cout << ".....printing out joints_2d_orig\n";
        // for(auto i = 0; i < joints_2d_orig.size(); i++)
        // {
        //     std::cout << "i: " << i << " size of this: " << joints_2d_orig.at(i).size() << "\n";
        //     for(auto j = 0; j < joints_2d_orig.at(i).size(); j++)
        //     {
        //         for(auto k = 0; k < joints_2d_orig.at(i).at(j).size(); k++)
        //         {
        //             std::cout << "j: " << j << ",k: " << k << " joint: " << joints_2d_orig.at(i).at(j).at(k) << "\n";
        //         }
        //     }
        // }

        // std::cout << "----------reading snowmen----------\n";
        for(auto sm = 0; sm < snowmen.size(); sm++)
        {
            // std::cout << " <0o0E snowman: " << sm << "\n";
            // std::cout << " cen_x: " << snowmen.at(sm).at(0) << "\n";

            int cen_x = snowmen.at(sm).at(0);
            float top_of_head = snowmen.at(sm).at(1);
            float bot_of_head = snowmen.at(sm).at(2);
            float top_of_mid = snowmen.at(sm).at(3);
            float bot_of_mid = snowmen.at(sm).at(4);
            float top_of_leg = snowmen.at(sm).at(5);
            float bot_of_leg = snowmen.at(sm).at(6);

            // does not guarantee attached snowman because poseKeypoints[{}] give float, so we compute with decimal points
            //      but image pixels only handle int values
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

            cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

            cv::Point mid_bot((int)cen_x, (int)bot_of_leg);
            cv::Point right_bot((int)cen_x + rad_bot, (int)bot_of_leg);
            cv::Point left_bot((int)cen_x - rad_bot, (int)bot_of_leg);

            // std::cout << "=======> left_bot: " << left_bot << "\n";

            cv::Point left_top((int)cen_x - rad_bot, (int)top_of_head);
            cv::Point right_top((int)cen_x + rad_bot, (int)top_of_head);
            
            cv::line(currPose_output, mid_bot, right_bot, cv::Scalar(255,0,0), 1, 8, 0);
            cv::line(currPose_output, mid_bot, left_bot, cv::Scalar(255,0,0), 1, 8, 0);

            cv::line(currPose_output, left_bot, left_top, cv::Scalar(255,0,0), 1, 8, 0);
            cv::line(currPose_output, right_bot, right_top, cv::Scalar(255,0,0), 1, 8, 0);

            cv::line(currPose_output, left_top, right_top, cv::Scalar(255,0,0), 1, 8, 0);

            cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);
            cv::line(currPose_output, image_center, left_bot, cv::Scalar(0,0,0), 1, 8, 0);
            
            std::vector<float> snowman_temp = snowmen.at(sm);
            snowman_temp.push_back(left_bot.x); // 7. left_bot.x
            snowman_temp.push_back(left_bot.y); // 8. left_bot.y

            // Sort snowman from floor level
            if(bot_of_leg > lowest_snowman)
            {
                // std::cout << "--------------FIND A PLACE TO INSERT CURRENT SNOWMAN-----------------\n";
                // if(snowmen_sorted.empty())
                // {
                //     std::cout << "--FIRST TIME IN sm = 0--\n";
                //     snowmen_sorted.push_back(snowmen.at(sm));
                // }
                // else
                // {
                //     std::cout << "--IN ELSE--\n";
                    snowmen_sorted.insert(snowmen_sorted.begin(), snowman_temp);
                    joints_3d.insert(joints_3d.begin(), datumsPtr_joints_3d.at(sm));
                    joints_2d.insert(joints_2d.begin(), joints_2d_orig.at(sm));
                // }

                lowest_snowman = bot_of_leg;
            }
            else 
            {
                // std::cout << "--------------ELSE FIND A PLACE TO INSERT CURRENT SNOWMAN-----------------\n";
                int s = 0;
                while(true)
                {
                    // std::cout << "s: " << s << "\n";
                    if(s >= snowmen_sorted.size()) break;
                    if(bot_of_leg > snowmen_sorted.at(s).at(6)) break;
                    s++;
                }

                // std::cout << "s out: " << s << "\n";
                snowmen_sorted.insert(snowmen_sorted.begin() + s, snowman_temp);
                joints_3d.insert(joints_3d.begin() + s, datumsPtr_joints_3d.at(sm));
                joints_2d.insert(joints_2d.begin() + s, joints_2d_orig.at(sm));
            }

        }

        datumsPtr->joints_3d_root_relative.clear();
        datumsPtr->joints_3d_root_relative = joints_3d;
        // draw center of image
        // std::cout << "currPose_output rows: " << currPose_output.rows << "\n";
        // std::cout << "currPose_output cols: " << currPose_output.cols << "\n";
        // std::cout << "currPose_output rows/2: " << currPose_output.rows/2 << "\n";
        // std::cout << "currPose_output cols/2: " << currPose_output.cols/2 << "\n";
        // cv::circle(currPose_output, cv::Point((int) currPose_output.cols/2, (int) currPose_output.rows/2), 5, cv::Scalar(0,255,0), -1, 8);

        // float poses_relations[snowmen.size()][snowmen.size()];
        std::vector<float> poses_relations; // store relative depths, may extend to storing relatiive heights

        // std::cout << "snowmen_sorted size: " << snowmen_sorted.size() << "\n";
        // std::cout << "................CHECKING snowmen_sorted................\n";
        for(auto s = 0; s < snowmen_sorted.size() - 1; s++)
        {
            // std::cout << "=======^OoO))s snowman: " << s << "\n";
            
            // int cen_x = snowmen_sorted.at(s).at(0);
            // float top_of_head = snowmen_sorted.at(s).at(1);
            // float bot_of_head = snowmen_sorted.at(s).at(2);
            // float top_of_mid = snowmen_sorted.at(s).at(3);
            // float bot_of_mid = snowmen_sorted.at(s).at(4);
            // float top_of_leg = snowmen_sorted.at(s).at(5);
            // float bot_of_leg = snowmen_sorted.at(s).at(6);
            float x_left_bot_a = snowmen_sorted.at(s).at(7);
            float y_left_bot_a = snowmen_sorted.at(s).at(8);

            // std::cout << "cen_x: " << snowmen_sorted.at(s).at(0) << "\n";
            // std::cout << "top_of_head: " << top_of_head << "\n";
            // std::cout << "bot_of_head: " << bot_of_head << "\n";
            // std::cout << "top_of_mid: " << top_of_mid << "\n";
            // std::cout << "bot_of_mid: " << bot_of_mid << "\n";
            // std::cout << "top_of_leg: " << top_of_leg << "\n";
            // std::cout << "bot_of_leg: " << bot_of_leg << "\n";

            // std::cout << "left_bot: (" << snowmen_sorted.at(s).at(7) << "," << snowmen_sorted.at(s).at(8) << ")\n";


            // FIND LENGTH FROM VANISHING POINT TO LEFT_BOT
            cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);
            float length_a = pow(pow((image_center.x - x_left_bot_a),2) + pow((image_center.y - y_left_bot_a),2), 0.5);
            // std::cout << "image_center: " << image_center << "\n";
            // std::cout << "length_a: " << length_a << "\n"; 
            // std::cout << "=====================\n";

            // cv::circle(currPose_output, image_center, 5, cv::Scalar(255,255,0), -1 ,8);
            // cv::circle(currPose_output, cv::Point((int) x_left_bot_a, (int) y_left_bot_a), 5, cv::Scalar(0,255,0), -1 ,8);


            // snowman s+1
            // std::cout << "=======^OoO))s snowman+1: " << s+1 << "\n";
            float x_left_bot_b = snowmen_sorted.at(s+1).at(7);
            float y_left_bot_b = snowmen_sorted.at(s+1).at(8);

            // std::cout << "cen_x: " << snowmen_sorted.at(s+1).at(0) << "\n";
            // std::cout << "top_of_head: " << top_of_head << "\n";
            // std::cout << "bot_of_head: " << bot_of_head << "\n";
            // std::cout << "top_of_mid: " << top_of_mid << "\n";
            // std::cout << "bot_of_mid: " << bot_of_mid << "\n";
            // std::cout << "top_of_leg: " << top_of_leg << "\n";
            // std::cout << "bot_of_leg: " << bot_of_leg << "\n";

            // std::cout << "left_bot: (" << snowmen_sorted.at(s+1).at(7) << "," << snowmen_sorted.at(s+1).at(8) << ")\n";


            // FIND LENGTH FROM VANISHING POINT TO LEFT_BOT
            float length_b = pow(pow((image_center.x - x_left_bot_b),2) + pow((image_center.y - y_left_bot_b),2), 0.5);
            // std::cout << "image_center: " << image_center << "\n";
            // std::cout << "length_b: " << length_b << "\n"; 
            // std::cout << "=====================\n";

            cv::circle(currPose_output, image_center, 5, cv::Scalar(255,255,0), -1 ,8);
            cv::circle(currPose_output, cv::Point((int) x_left_bot_b, (int) y_left_bot_b), 5, cv::Scalar(0,255,0), -1 ,8);

            float x0 = image_center.x; // x of image center
            float y0 = image_center.y; // y of image center
            float x1 = x_left_bot_a; // x of larger snowman left bot
            float y1 = y_left_bot_a; // y of larger snowman left bot
            float y = y_left_bot_b; // y of src
            float x = (((y-y0)*(x1-x0))/(y1-y0)) + x0;

            // std::cout << "=========================\n=========================\n";
            // std::cout << "x0: " << x0 << ", y0: " << y0 << "\n";
            // std::cout << "x1: " << x1 << ", y1: " << y1 << "\n";
            // std::cout << "y: " << y << "\n"; 

            // std::cout << "--> interpolated x = " << x << "(int=" << (int)x << ")\n";
                    
            cv::Point start_interpolate((int) x_left_bot_b, (int) y_left_bot_b);
            cv::Point end_interpolate((int) x, (int) y_left_bot_b);
            cv::line(currPose_output, start_interpolate, end_interpolate, cv::Scalar(0,0,0), 1, 8, 0);
            cv::circle(currPose_output, end_interpolate, 3, cv::Scalar(0,0,0), -1, 8);

            // FIND INTERPOLATED LENGTH
            float length_x = pow(pow((image_center.x - x),2) + pow((image_center.y - y_left_bot_b),2), 0.5);

            std::cout << "length_x: " << length_x << "\n";

            // NORMALISE LENGTH AND GET EXPONENT SCALE
            float norm_factor = 1/length_a;
            float norm_scale = 5;
            float norm_length = abs(length_a - length_x) * norm_factor * norm_scale;
            float exp = 2.71828;
            float infer_depth = pow((10*exp),-(5 - norm_length));
            // std::cout << "5 - norm_length: " << 5 - norm_length << "\n";
            // std::cout << "norm_length: " << norm_length << "\n";
            // std::cout << "infer_depth: " << infer_depth*10000000 << "\n";
            datumsPtr->floorLevels.push_back(infer_depth);
        }

        std::cout << "snowmen_sorted size(): " << snowmen_sorted.size() << "\n";
        std::vector<std::vector<float>> prevSnowmen = datumsPtr->prevSnowmen;

        std::vector<float> score_stage1;
        // datumsPtr->snowmen.clear();
        std::cout << "......................GOING INTO PAIRING SNOWMEN-------------------------\n";
        for(auto i = 0; i < snowmen_sorted.size(); i++)
        {
            std::cout << "i: " << i << "\n";
            // std::cout << "==> x chest: " << joints_2d.at(i).at(14).at(0) << "\n";
            // std::cout << "==> y chest: " << joints_2d.at(i).at(14).at(1) << "\n";
            datumsPtr->snowmen.push_back(snowmen_sorted.at(i));
            datumsPtr->orientation.push_back(joints_2d.at(i).at(14).at(0)); // x chest
            datumsPtr->orientation.push_back(joints_2d.at(i).at(14).at(1)); // y chest

            // std::cout << "snowmen_sorted i size: " << snowmen_sorted.at(i).size() << "\n";

            float cen_x = snowmen_sorted.at(i).at(0);
            float top_of_head = snowmen_sorted.at(i).at(1);
            float bot_of_head = snowmen_sorted.at(i).at(2);
            float top_of_mid = snowmen_sorted.at(i).at(3);
            float bot_of_mid = snowmen_sorted.at(i).at(4);
            float top_of_leg = snowmen_sorted.at(i).at(5);
            float bot_of_leg = snowmen_sorted.at(i).at(6);
            // float x_left_bot_a = snowmen_sorted.at(s).at(7);
            // float y_left_bot_a = snowmen_sorted.at(s).at(8);

            float rad_top =  bot_of_head - (top_of_head + bot_of_head) / 2;
            float rad_mid =  bot_of_mid - (top_of_mid + bot_of_mid) / 2;
            float rad_bot = bot_of_leg - (top_of_leg + bot_of_leg) / 2;

            // std::cout << "\tcen_x: " << cen_x << "\n";
            // std::cout << "\ttop radius: " << rad_top << "\n";
            // std::cout << "\tmid radius: " << rad_mid << "\n";
            // std::cout << "\tbot radius: " << rad_bot << "\n";

            // COMPARE WITH ALL PREV SNOWMEN
            for(auto p = 0; p < prevSnowmen.size(); p++)
            {
                float cen_x_p = prevSnowmen.at(p).at(0);
                float top_of_head_p = prevSnowmen.at(p).at(1);
                float bot_of_head_p = prevSnowmen.at(p).at(2);
                float top_of_mid_p = prevSnowmen.at(p).at(3);
                float bot_of_mid_p = prevSnowmen.at(p).at(4);
                float top_of_leg_p = prevSnowmen.at(p).at(5);
                float bot_of_leg_p = prevSnowmen.at(p).at(6);
                // float x_left_bot_a_p = prevSnowmen.at(s).at(7);
                // float y_left_bot_a_p = prevSnowmen.at(s).at(8);

                float rad_top_p =  bot_of_head_p - (top_of_head_p + bot_of_head_p) / 2;
                float rad_mid_p =  bot_of_mid_p - (top_of_mid_p + bot_of_mid_p) / 2;
                float rad_bot_p = bot_of_leg_p - (top_of_leg_p + bot_of_leg_p) / 2;
                std::cout << "...................COMPARING WITH ALL PREV SNOWMEN----\n";
                std::cout << "  .p: " << p << ", size(): " << prevSnowmen.at(p).size() << "\n";
                // std::cout << "      cen_x_p: " << cen_x_p << "\n";
                // std::cout << "      top radius: " << rad_top_p << "\n";
                // std::cout << "      mid radius: " << rad_mid_p << "\n";
                // std::cout << "      bot radius: " << rad_bot_p << "\n";

                // std::cout << "cen_x diff: " << cen_x - cen_x_p << "\n";
                // std::cout << "top rad diff: " << rad_top - rad_top_p << "\n";
                // std::cout << "mid rad diff: " << rad_mid - rad_mid_p << "\n";
                // std::cout << "bot rad diff: " << rad_bot - rad_bot_p << "\n";

                float diff_cen = cen_x - cen_x_p;
                float diff_top = rad_top - rad_top_p;
                float diff_mid = rad_mid - rad_mid_p;
                float diff_bot = rad_bot - rad_bot_p;

                if(diff_top == 0 && diff_mid == 0 && diff_bot == 0)
                {
                    std::cout << "----------------got exact same pose-----------------\n";
                }

                float score = (0.3 * diff_cen) + (0.3 * diff_mid) + (0.2 * diff_top) + (0.1 * diff_bot);
                std::cout << "->score: " << score << "\n";
                score_stage1.push_back(score == 0 ? 10 : abs(1/score)); // not considering direction

            }

        }

        // for(auto sc = 0; sc < score_stage1.size(); sc++)
        // {
        std::cout << "||||||||||||||||||SCORE STAGE 1||||||||||||||||||\n";
        std::vector<int> scstage1_indices;
        auto score_index = 0;
            for(auto c = 0; c < snowmen_sorted.size(); c++)
            {
                std::cout << "curr snowmen: " << c << "\n";

                std::vector<float> score_stage1_temp;
                for(auto p = 0; p < prevSnowmen.size(); p++)
                {
                    std::cout << "->sc: " << score_index << ", score: " << score_stage1.at(score_index) << "\n";
                    score_stage1_temp.push_back(score_stage1.at(score_index));
                    score_index++;
                }
                // std::cout << "-----------check 1---------\n";
                float score_max_temp = vNectFindMax(score_stage1_temp);
                // std::cout << "-----------check 2---------\n";
                int score_max_index = getMaxIndex(score_stage1_temp, score_max_temp);
                scstage1_indices.push_back(score_max_index);
                // std::cout << "-----------check 3---------\n";
            }
        std::cout << "---------\n";
        // std::cout << "---> prevSnowmen.at(scstage1_indices.at(i)): " << prevSnowmen.at(0).at(0) << "\n";  
        
        if(!prevSnowmen.empty())
        {
            for(auto i = 0; i < scstage1_indices.size(); i++)
            {
                std::cout << "i: " << i << ", " << scstage1_indices.at(i) << "\n";
                std::cout << "  global index: " << (i * (prevSnowmen.size())) + scstage1_indices.at(i) << "\n";

                // std::cout << "---> prevSnowmen.at(scstage1_indices.at(i)): " << prevSnowmen.at(0).at(0) << "\n";

                int cen_x = prevSnowmen.at((int)scstage1_indices.at(i)).at(0);
                // std::cout << "----->check 1\n";
                float top_of_head = prevSnowmen.at(scstage1_indices.at(i)).at(1);
                float bot_of_head = prevSnowmen.at(scstage1_indices.at(i)).at(2);
                float top_of_mid = prevSnowmen.at(scstage1_indices.at(i)).at(3);
                float bot_of_mid = prevSnowmen.at(scstage1_indices.at(i)).at(4);
                float top_of_leg = prevSnowmen.at(scstage1_indices.at(i)).at(5);
                float bot_of_leg = prevSnowmen.at(scstage1_indices.at(i)).at(6);
                float x_left_bot_a = prevSnowmen.at(scstage1_indices.at(i)).at(7);
                float y_left_bot_a = prevSnowmen.at(scstage1_indices.at(i)).at(8);

                // std::cout << "----->check 2\n";
                float rad_top =  bot_of_head - (top_of_head + bot_of_head) / 2;
                float rad_mid =  bot_of_mid - (top_of_mid + bot_of_mid) / 2;
                float rad_bot = bot_of_leg - (top_of_leg + bot_of_leg) / 2;
                
                // std::cout << "----->check 3\n";
                // DRAW SNOWMAN ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                cv::circle(currPose, cv::Point(cen_x, (int) (top_of_head + bot_of_head)/2), rad_top, cv::Scalar(0,255,0), -1, 8);
                cv::circle(currPose, cv::Point(cen_x, (int) (top_of_mid + bot_of_mid)/2), rad_mid, cv::Scalar(0,255,0), -1, 8);
                cv::circle(currPose, cv::Point(cen_x, (int) (top_of_leg + bot_of_leg)/2), rad_bot, cv::Scalar(0,255,0), -1, 8);

                // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)top_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)top_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_head), 5, cv::Scalar(0,255,255), -1, 8);
                // cv::circle(currPose, cv::Point(cen_x, (int)top_of_head), 5, cv::Scalar(0,255,255), -1, 8);

                cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
                ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            }
        }

        std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||\n";
        // }

        // auto size_orient;
        // if(datumsPtr->prevOrientation.size() > datumsPtr->orientation.size()) size_orient = datumsPtr->orientation.size();
        // else if(datumsPtr->prevOrientation.size() <= datumsPtr->orientation.size()) size_orient = datumsPtr->prevOrientation.size();

        for(auto c = 0; c < datumsPtr->orientation.size(); c++)
        {
            if(datumsPtr->orientation.at(c) == 0) std::cout << "orientation is zero!!\n";
            for(auto p = 0; p < datumsPtr->prevOrientation.size(); p++)
            {
                // std::cout << "i: " << i << "prevOri: " << datumsPtr->prevOrientation.at(i) << "\n";
                if(datumsPtr->prevOrientation.at(p) == 0) std::cout << "prevOrientation is zero!!\n";
            }

        }

        // ----------------------------------- GET DISTANCES BETWEEN EACH PAIR POSE-------------------------------- //
        for (int person = 0 ; person < joints_2d.size() - 1 ; person++)
        {

            // float neckDiff = (poseKeypoints[{person+1, 1, 0}] - poseKeypoints[{person, 1, 0}]);
            float neckDiff = joints_2d.at(person+1).at(1).at(0) - joints_2d.at(person).at(1).at(0);
            // float chestDiff = (poseKeypoints[{person+1, 14, 0}] - poseKeypoints[{person, 14, 0}]);
            float chestDiff = joints_2d.at(person+1).at(14).at(0) - joints_2d.at(person).at(14).at(0);

            // float lowestAnkle_1 = poseKeypoints[{person, 13, 1}] > poseKeypoints[{person, 10, 1}] ? poseKeypoints[{person, 13, 1}] : poseKeypoints[{person, 10, 1}];  // y values of {LAnkle: 13}, {RAnkel: 10}
            // float lowestAnkle_2 = poseKeypoints[{person+1, 13, 1}] > poseKeypoints[{person+1, 10, 1}] ? poseKeypoints[{person+1, 13, 1}] : poseKeypoints[{person+1, 10, 1}];

            // float ankleRatio = lowestAnkle_1 > lowestAnkle_2 ? (lowestAnkle_1 - lowestAnkle_2) : (lowestAnkle_2 - lowestAnkle_1);

            // std::cout << "person pair: [" << person << "-" << person+1 << "]\n";
            // std::cout << "neckDiff: " << neckDiff << "\n";
            // std::cout << "chestDiff: " << chestDiff << "\n"; 

            datumsPtr->neckDiffs.push_back(neckDiff);
            datumsPtr->chestDiffs.push_back(chestDiff);
            // poseDistances.push_back(neckDiff);
        }



        // "-----:::: jointsRad ::::-----" //
        // for(int i = 0; i < jointsRad.size(); i++)
        // {
        //     std::cout << "jointsRad i = " << i << "\n";
        //     for(int j = 0; j < jointsRad.at(i).size(); j++)
        //     {
        //         std::cout << "  " << j << ". " << jointsRad.at(i).at(j) << "\n";
        //     }
        // }

        // "-----:::: jointsCen ::::-----" //
        // for(int i = 0; i < jointsCen.size(); i++)
        // {
        //     std::cout << "jointsCen i = " << i << "\n";
        //     for(int j = 0; j < jointsCen.at(i).size(); j++)
        //     {
        //         std::cout << "  " << j << ". " << jointsCen.at(i).at(j) << "\n";
        //     }
        // }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //       std::vector<std::vector<cv::Point>> jointsCenExtr;

  //       // "-----:::: find extreme points at head, floor left and floor right ::::-----" //
  //       for(int i = 0; i < (int)jointsRad.size(); i++)
  //       {
  //           std::vector<cv::Point> currExtr;
  //           for(int j = 0; j < jointsRad.at(i).size(); j++)
  //           {   
  //               if(j == 0) //neck
  //               {
  //                   // float neck_top_x = jointsCen.at(i).at(j).x + jointsRad.at(i).at(j);
  //                   float neck_top_y = jointsCen.at(i).at(j).y - jointsRad.at(i).at(j);
  //                   currExtr.push_back(cv::Point(jointsCen.at(i).at(j).x, neck_top_y));
  //               }
  //               else if(j == 1)
  //               {
  //                   // float lknee_bottom_x = jointsCen.at(i).at(j).x + jointsRad.at(i).at(j);
  //                   float lknee_bottom_y = jointsCen.at(i).at(j).y + jointsRad.at(i).at(j);
  //                   currExtr.push_back(cv::Point(jointsCen.at(i).at(j).x, lknee_bottom_y));
  //               }
  //               else if(j == 2)
  //               {
  //                   float rknee_bottom_y = jointsCen.at(i).at(j).y + jointsRad.at(i).at(j);
  //                   currExtr.push_back(cv::Point(jointsCen.at(i).at(j).x, rknee_bottom_y));
  //               }
  //           }
  //           jointsCenExtr.push_back(currExtr);
  //       }

  //       for(int i = 0; i < (int)jointsCenExtr.size(); i++)
  //       {
  //           // find max length in height
  //           int maxHeight = jointsCenExtr.at(i).at(1).y > jointsCenExtr.at(i).at(2).y ? abs(jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(0).y) : abs(jointsCenExtr.at(i).at(2).y - jointsCenExtr.at(i).at(0).y);
  //           // std::cout << "jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(2).y ==> " << jointsCenExtr.at(i).at(1).y - jointsCenExtr.at(i).at(2).y << "\n";
  //           // for(int j = 0; j < jointsCenExtr.at(i).size(); j++)
  //           // {
  //           //     std::cout << j << ". " << jointsCenExtr.at(i).at(j) << "\n";
  //               cv::circle(currPose_output, jointsCenExtr.at(i).at(1), 5, cv::Scalar(0,255,255), -1, 8);
  //               cv::circle(currPose_output, jointsCenExtr.at(i).at(2), 5, cv::Scalar(0,180,255), -1, 8);

  //               //find which ankle is lower
  //               float lower_ankle_height = jointsCenExtr.at(i).at(1).y > jointsCenExtr.at(i).at(2).y ? jointsCenExtr.at(i).at(1).y : jointsCenExtr.at(i).at(2).y;
  //               float higher_ankle_height = jointsCenExtr.at(i).at(1).y < jointsCenExtr.at(i).at(2).y ? jointsCenExtr.at(i).at(1).y : jointsCenExtr.at(i).at(2).y;

  //               //find 5% tolerance bound within lower_ankle_height
  //               float five_percent = (lower_ankle_height * 0.05);
  //               float upper_bound = abs(lower_ankle_height - five_percent);
  //               float lower_bound = abs(lower_ankle_height + five_percent);

  //               // if higher_ankle_height is within 5% of lower ankle_height, average the floor level height
  //               if(upper_bound < higher_ankle_height < lower_bound)
  //               {
  //                   int avg_x = (int)(jointsCenExtr.at(i).at(1).x + jointsCenExtr.at(i).at(2).x)/2;
  //                   int avg_y = (int)(jointsCenExtr.at(i).at(1).y + jointsCenExtr.at(i).at(2).y)/2;
  //                   cv::circle(currPose_output, cv::Point(jointsCenExtr.at(i).at(0).x, avg_y),5,cv::Scalar(180,180,180),-1,8);
  //               }


  //           // }
  //       }
	
		// // "------------- CHECK JOINTS RAD AND CEN -------------" //
  //       std::vector<std::vector<float>> jointsChose;
  //       std::vector<std::vector<cv::Point>> jointsChosePoints;
		// for(int i = 0; i < (int)jointsRad.size(); i++)
  //       {            
  //           std::vector<float> choPart;
  //           std::vector<cv::Point> choPoints;

  //           float top = jointsRad.at(i).at(0) ;
  //           float bot = jointsRad.at(i).at(1) > jointsRad.at(i).at(2) ? jointsRad.at(i).at(1) : jointsRad.at(i).at(2);
  //           float mid = jointsRad.at(i).at(3) > jointsRad.at(i).at(4) ? jointsRad.at(i).at(3) : jointsRad.at(i).at(4);

  //           cv::Point topPoint = jointsCen.at(i).at(0);
  //           cv::Point botPoint = jointsRad.at(i).at(1) > jointsRad.at(i).at(2) ? jointsCen.at(i).at(1) : jointsCen.at(i).at(2);
  //           cv::Point midPoint = jointsRad.at(i).at(3) > jointsRad.at(i).at(4) ? jointsCen.at(i).at(3) : jointsCen.at(i).at(4);

  //           choPart.push_back(top);
  //           choPart.push_back(bot);
  //           choPart.push_back(mid);

  //           choPoints.push_back(topPoint);
  //           choPoints.push_back(botPoint);
  //           choPoints.push_back(midPoint);
            
  //           jointsChose.push_back(choPart);
  //           jointsChosePoints.push_back(choPoints);
  //       }        

  //       // "------------- JOINTS AVG -------------" //
  //       std::vector<float> poseFloorLevel; 
  //       for(int i = 0; i < (int)jointsChose.size(); i++)
  //       {
  //           // std::cout << "i: " << i << "\n";
  //           // std::cout << "floor: " << jointsChosePoints.at(i).at(1).y + jointsChose.at(i).at(1) << "\n";
  //           float floorLevel = jointsChosePoints.at(i).at(1).y + jointsChose.at(i).at(1);
  //           poseFloorLevel.push_back(floorLevel);

  //           // cv::line(currPose_output, cv::Point(jointsChosePoints.at(i).at(1).x, floorLevel), cv::Point(jointsChosePoints.at(i).at(1).x + 10, floorLevel + 20), cv::Scalar(255,0,0), 1, 8, 0);
  //           // std::string text = "(" + std::to_string(jointsChosePoints.at(i).at(1).x) + "," + std::to_string(jointsChosePoints.at(i).at(1).y) + ")";
  //           // cv::Point textOrg = cv::Point(jointsChosePoints.at(i).at(1).x + 10, floorLevel + 35);
  //           // int fontFace = CV_FONT_HERSHEY_SIMPLEX;
  //           // cv::putText(currPose_output, text, textOrg, fontFace, 0.3, cv::Scalar(255,0,255), 1, 8);

  //           // for(int j = 0; j < jointsChose.at(i).size(); j++)
  //           // {
  //           //     // std::cout << jointsChose.at(i).at(j) << "\n";
  //           //     std::cout << jointsChosePoints.at(i).at(j) << "\n";
  //           // }
  //       }

		// // for now, kind of assuming every pose has the same height or whatever height given by VNect
		// for(int i = 0; i < (int)poseFloorLevel.size()-1; i++)
		// {
		// 	std::cout << "floor level difference: " << (float) abs(poseFloorLevel.at(i) - poseFloorLevel.at(i+1)) << "\n";
		// 	datumsPtr->floorLevels.push_back((float) abs(poseFloorLevel.at(i) - poseFloorLevel.at(i+1)));
		// }
		// // compare radius
		// // depth depends on poses interdependency positions mainly, not where there are in the original image

		// cv::imshow("currPose_output", currPose_output);


    }


    std::vector<std::vector<float>> vNectForward(cv::Mat croppedImg)
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

	        cv::copyMakeBorder(sample, sample, top, bottom, start, start - remainder_src , cv::BORDER_CONSTANT, 0);
	        // cv::imshow("sample less than input network", sample);
	        // std::cout << "sample size: " << sample.size() << "\n";
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
	    // int end_v2 = (half + half_v2) + remainder_v2;

	    int length_v3 = sample_resized_v3.rows;
	    int half_v3 = std::floor(length_v3/2);
	    int remainder_v3 = length_v3 % 2;

	    int start_v3 = (half - half_v3);
	    // int end_v3 = (half + half_v3) + remainder_v3;

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
	    for(int i = 0; i < (int)imgs.size(); i++)
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
	            default: std::cout << "VNECT ERROR!!! HMS OUTPUT OUT OF BOUND!!!"; break;
	        }
	        outputblobi++;
	    }

	    /* ------------- AVERAGE THE HEATMAPS OUT -------------  */
	    int nums_out = testnet->output_blobs().at(0)->shape(0);
	    int channels_out = testnet->output_blobs().at(0)->shape(1);
	    // int height_out = testnet->output_blobs().at(0)->shape(2);
	    // int width_out = testnet->output_blobs().at(0)->shape(3);

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


	    for(int i = 0; i < (int)max_locs.size(); i++)
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

	    for(int i = 0; i < (int)max_locs.size(); i++)
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

	    for(int i = 0; i < (int)joints_2d.size(); i++)
	    {
	        cv::circle(image_2d, joints_2d.at(i), 3, cv::Scalar(0,0,255), -1, 8);
            // right ankle
            // if(i == 10) {
            //     cv::circle(image_2d, joints_2d.at(i), 5, cv::Scalar(255,0,255), -1, 8);
            // }
            // left ankle
            // if(i == 13) {
            //     cv::circle(image_2d, joints_2d.at(i), 5, cv::Scalar(255,255,0), -1, 8);
            // }
	    }

	    // cv::imshow(std::to_string(rand() % 23), image_2d);

		/*---- PRINT VNECT_3D_JOINTS.TXT ----*/
		// // "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/3d_joints/"
		// std::string saveToName = pathToWrite + croppedImgName + ".txt";
		// std::ofstream out3djoints(saveToName);
		// std::streambuf *coutbuf3djoints = std::cout.rdbuf();
		// std::cout.rdbuf(out3djoints.rdbuf());

		// // int joint_i = 0;
		// for(auto joint : joints_3d_root_relative)
		// {
		// 	// if(joint_i == 15) break;
		// 	std::cout << joint.at(0) << " ";
		// 	std::cout << joint.at(1) << " ";
		// 	std::cout << joint.at(2) << " ";
		// 	std::cout << "\n";
		// 	// joint_i++;
		// }

		// std::cout.rdbuf(coutbuf3djoints);

		// ----- TESTING draw3DPython ----- //
		// std::vector<std::string> fileNames;
		// draw3DPython(fileNames);

		return joints_3d_root_relative;
    }
}