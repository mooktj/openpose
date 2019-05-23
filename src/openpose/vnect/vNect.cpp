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
#include <openpose/vnect/snowMan.hpp>

namespace op
{
    template <class valType>
    std::vector<valType> getOccIndices(std::vector<valType> in, valType val)
    {
        std::vector<valType> out;
        for(auto i = 0; i < in.size(); i++)
        {
            if(val == in.at(i))
            {
                out.push_back(i);
            }
        }
        return out;
    }

    // template <class valType>
   	float vNectFindMax(std::vector<float> in)
    {
        float max = 0;
        for(auto i = 0; i < in.size(); i++)
        {
            if(max < in.at(i)) max = in.at(i);
        }
        return max;
    }

    template <class valType>
    int getVecIndex(std::vector<valType> in, valType val, int orderReq)
    {
        // std::cout << "~~~~~~ :: getMaxIndex :: ~~~~~~\n";
        // float max = 0;
        // std::cout << "val: " << val << "\n";
        auto orderFound = 0;
        bool FOUND = false;
        auto i = 0;
        for(i = 0; i < in.size(); i++)
        {
            // std::cout << "in.at(" << i << "): " << in.at(i) << "\n";
            if(val == in.at(i)) 
                {   
                    if(orderFound == orderReq) {
                        FOUND = true;
                        break;
                    } else
                    {
                        orderFound++;
                    }
                }
        }

        // std::cout << "getMaxIndex: " << i << "\n";
        return FOUND ? (int) i : -1;
    }

    // template <class valType>
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
        std::vector<std::vector<float>> snowmen;

		cv::Mat currPose = datumsPtr->cvOutputData.clone();
		cv::Mat currPose_output = datumsPtr->cvOutputData.clone();
		float opacity = 0.5;

        // ---------- PRE-STAGE 1: TO GET ALL POINTS FOR ALL DETECTED POSEKEYPOINTS TO BUILD SNOWMEN ---------- //
        // std::cout << "------------------------PREPARE CURR SNOWMEN-------------------------\n";
		for(int i = 0; i < poseKeypoints.getSize(0); i++)
        {
            std::cout << "----------pose: " << i << "\n";
            std::vector<float> snowman;

            // ----- ALL REQUIRED JOINTS ----- //
            float x_head = poseKeypoints[{i,0,0}];
            float y_head = poseKeypoints[{i,0,1}];

            float x_neck = poseKeypoints[{i,1,0}];
            float y_neck = poseKeypoints[{i,1,1}];

            float x_chest = poseKeypoints[{i,14,0}];
            float y_chest = poseKeypoints[{i,14,1}];

            float x_lhip = poseKeypoints[{i,11,0}];
            float y_lhip = poseKeypoints[{i,11,1}];

            float x_rhip = poseKeypoints[{i,8,0}];
            float y_rhip = poseKeypoints[{i,8,1}];

            float x_lknee = poseKeypoints[{i,12,0}];
            float y_lknee = poseKeypoints[{i,12,1}];

            float x_rknee = poseKeypoints[{i,9,0}];
            float y_rknee = poseKeypoints[{i,9,1}];

            float x_lankle = poseKeypoints[{i,13,0}];
            float y_lankle = poseKeypoints[{i,13,1}];

            float x_rankle = poseKeypoints[{i,10,0}];
            float y_rankle = poseKeypoints[{i,10,1}];

            // std::cout << "x_head: " << x_head << "\n";
            // std::cout << "y_head: " << y_head << "\n";

            // std::cout << "x_neck: " << x_neck << "\n";
            // std::cout << "y_neck: " << y_neck << "\n";

            // std::cout << "x_chest: " << x_chest << "\n";
            // std::cout << "y_chest: " << y_chest << "\n";

            // std::cout << "x_lhip: " << x_lhip << "\n";
            // std::cout << "y_lhip: " << y_lhip << "\n";

            // std::cout << "x_rhip: " << x_rhip << "\n";
            // std::cout << "y_rhip: " << y_rhip << "\n";

            // std::cout << "x_lankle: " << x_lankle << "\n";
            // std::cout << "y_lankle: " << y_lankle << "\n";

            // std::cout << "x_rankle: " << x_rankle << "\n";
            // std::cout << "y_rankle: " << y_rankle << "\n";

            // std::cout << "x_lknee: " << x_lknee << "\n";
            // std::cout << "y_lknee: " << y_lknee << "\n";

            // std::cout << "x_rknee: " << x_rknee << "\n";
            // std::cout << "y_rknee: " << y_rknee << "\n";


            ////////////////////////////----TOP----////////////////////////////////////////////////////////////////

            float l_head = (x_neck == 0 || x_head == 0) || (y_neck == 0 || y_head == 0) ? 0 : pow(pow((x_head - x_neck),2) + pow((y_head - y_neck),2), 0.5);
            float l_chest = (x_neck == 0 || x_chest == 0) || (y_neck == 0 || y_chest == 0) ? 0 : pow(pow((x_chest - x_neck),2) + pow((y_chest - y_neck),2), 0.5);

            // choose larger radius between head and chest
            float rad_top = l_head > l_chest ? l_head : l_chest;

            cv::Point cen_top = {(int)x_neck,(int)y_neck};

            // USE PREV DETECTED JOINTS IF INSUFFICIENT JOINTS DETECTED TO MAKE A TOP CIRCLE
            if( ((x_neck == 0 && x_head == 0) || (x_neck == 0 && x_chest == 0) || (x_head == 0 && x_chest == 0)) ||
                    ((y_neck == 0 && y_head == 0) || (y_neck == 0 && y_chest == 0) || (y_head == 0 && y_chest == 0)) )
            {
              // GET PREV
                // SET RADIUS TO 0 TO NOTIFY BELOW TO CHECK FOR RETRIEVING UNMATCHED POSES
                rad_top = 0;
                cen_top = {0,0};
            }

            // std::cout << "rad_top: " << rad_top << "\n";
            // std::cout << "cen_top: " << cen_top << "\n";

            ////////////////////////////----MID----////////////////////////////////////////////////////////////////

            float dia_lmid = (x_chest == 0 || x_lhip == 0) || (y_chest == 0 || y_lhip == 0) ? 0 : pow(pow((x_chest - x_lhip),2) + pow((y_chest - y_lhip),2), 0.5); // diameter of chest/lhip
            float rad_lmid = dia_lmid/2;
            cv::Point cen_lmid = {(int)(x_chest + x_lhip)/2, (int)(y_chest + y_lhip)/2};

            if( (x_chest == 0 && x_lhip == 0) || (y_chest == 0 && y_lhip == 0) )
            {
                // SET 0 TO NOTIFY UNDETECTED JOINTS, REQUIRE PREV POSE WHEN POST-MATCHING
                rad_lmid = 0;
                dia_lmid = 0;
                cen_lmid = {0,0};
            }

            float dia_rmid = (x_chest == 0 || x_rhip == 0) || (y_chest == 0 || y_rhip == 0) ? 0: pow(pow((x_chest - x_rhip),2) + pow((y_chest - y_rhip),2), 0.5); // diameter of chest/rhip
            float rad_rmid = dia_rmid/2;
            cv::Point cen_rmid = {(int)(x_chest + x_rhip)/2, (int)(y_chest + y_rhip)/2};
            
            if( (x_chest == 0 && x_rhip == 0) || (y_chest == 0 && y_rhip == 0))
            {
                rad_rmid = 0;
                dia_rmid = 0;
                cen_rmid = {0,0};
            }

            float dia_mid;
            int cen_x;
            if(dia_lmid == 0 && dia_rmid == 0) 
            {
                // NOTIFY TO GET PREV POSE
                dia_mid = 0;
                cen_x = 0;
            } 
            else if(dia_lmid == 0 && dia_rmid != 0) 
            {
                // JUST USE THE ONE COMPLETED
                dia_mid = dia_rmid; 
                cen_x = cen_rmid.x;
            }
            else if(dia_rmid == 0 && dia_lmid != 0)
            {
                // JUST USE THE ONE COMPLETED
                dia_mid = dia_lmid; 
                cen_x = cen_lmid.x;
            }
            else 
            {
                dia_mid = dia_lmid > dia_rmid ? dia_lmid : dia_rmid;
                cen_x = (int)(cen_lmid.x + cen_rmid.x)/2;
            }

            // std::cout << "dia_mid: " << dia_mid << "\n";
            // std::cout << "cen_x: " << cen_x << "\n";

            ////////////////////////////----BOT----////////////////////////////////////////////////////////////////            

            float l_lhip = (x_lhip == 0 || x_lknee == 0) || (y_lhip == 0 || y_lknee == 0) ? 0 : pow(pow((x_lhip - x_lknee),2) + pow((y_lhip - y_lknee),2), 0.5);
            float l_lankle = (x_lankle == 0 || x_lknee == 0) || (y_lankle == 0 || y_lknee == 0) ? 0 : pow(pow((x_lankle - x_lknee),2) + pow((y_lankle - y_lknee),2), 0.5);

            float l_rhip = (x_rhip == 0 || x_rknee == 0) || (y_rhip == 0 || y_rknee == 0) ? 0 : pow(pow((x_rhip - x_rknee),2) + pow((y_rhip - y_rknee),2), 0.5);
            float l_rankle = (x_rankle == 0 || x_rknee == 0) || (y_rankle == 0 || y_rknee == 0) ? 0 : pow(pow((x_rankle - x_rknee),2) + pow((y_rankle - y_rknee),2), 0.5);

            // choosing larger radius for LEFT BOT
            float rad_lbot = l_lhip > l_lankle ? l_lhip : l_lankle;

            cv::Point cen_lbot = {(int)x_lknee, (int)y_lknee};

            if( ((x_lknee == 0 && x_lankle == 0) || (x_lknee == 0 && x_lhip == 0) || (x_lankle == 0 && x_lhip == 0)) ||
                   ((y_lknee == 0 && y_lankle == 0) || (y_lknee == 0 && y_lhip == 0) || (y_lankle == 0 && y_lhip == 0)) )
            {
                // SET RAD_LBOT AND CEN_LBOT TO 0 TO NOTIFY IMPOSSIBLE CIRCLE
                std::cout << "^^^^ incomplete LEFT BOT ^^^^\n";
                rad_lbot = 0;
                cen_lbot = {0,0};
            }

            // choosing larger radius for RIGHT BOT
            float rad_rbot = l_rhip > l_rankle ? l_rhip : l_rankle;

            cv::Point cen_rbot = {(int)x_rknee, (int)y_rknee};

            if( ((x_rknee == 0 && x_rankle == 0) || (x_rknee == 0 && x_rhip == 0) || (x_rankle == 0 && x_rhip == 0)) ||
                   ((y_rknee == 0 && y_rankle == 0) || (y_rknee == 0 && y_rhip == 0) || (y_rankle == 0 && y_rhip == 0)) )
            {
                // SET RAD_RBOT AND CEN_RBOT TO 0 TO NOTIFY TO CHECK FOR UMATCHED POSES
                std::cout << "^^^^ incomplete RIGHT BOT ^^^^\n";
                rad_rbot = 0;
                cen_rbot = {0,0};
            }

            // can be different leg from lower_ankle_height because both legs are normally of the same/similar length
            float rad_bot = rad_lbot > rad_rbot ? rad_lbot : rad_rbot;
            float dia_leg = 2 * rad_bot;

            // std::cout << "rad_bot: " << rad_bot << "\n";
            // std::cout << "dia_leg: " << dia_leg << "\n";

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
            //------------GET TOP and BOT OF LEG-------------//
            float lower_ankle_height = y_lankle > y_rankle ? y_lankle : y_rankle;

            float bot_of_leg = lower_ankle_height;
            float top_of_leg = bot_of_leg - dia_leg;

            //------------GET TOP and BOT OF MID-------------//
            float bot_of_mid = top_of_leg;
            float top_of_mid = bot_of_mid - dia_mid;

            //------------GET TOP and BOT OF HEAD-------------//
            float bot_of_head = top_of_mid;
            float top_of_head = bot_of_head - (2 * rad_top);

            // std::cout << "bot_of_leg: " << bot_of_leg << "\n";
            // std::cout << "top_of_leg: " << top_of_leg << "\n";
            // std::cout << "bot_of_mid: " << bot_of_mid << "\n";
            // std::cout << "top_of_mid: " << top_of_mid << "\n";
            // std::cout << "bot_of_head: " << bot_of_head << "\n";
            // std::cout << "top_of_head: " << top_of_head << "\n";
            // std::cout << "cen_x: " << cen_x << "\n";

            // EXTRACT SNOWMAN
            snowman.push_back(float(cen_x));// 0. cen_x
            snowman.push_back(top_of_head); // 1. top_of_head
            snowman.push_back(bot_of_head); // 2. bot_of_head
            snowman.push_back(top_of_mid);  // 3. top_of_mid
            snowman.push_back(bot_of_mid);  // 4. bot_of_mid
            snowman.push_back(top_of_leg);  // 5. top_of_leg
            snowman.push_back(bot_of_leg);  // 6. bot_of_leg

            // std::cout << "----MAKING A SNOWMAN----\n";
            // std::cout << "cen_x: " << float(cen_x) << "\n"; // 0. cen_x
            // std::cout << "top_of_head: " << top_of_head << "\n"; // 1. top_of_head
            // std::cout << "bot_of_head: " << bot_of_head << "\n"; // 2. bot_of_head
            // std::cout << "top_of_mid: " << top_of_mid << "\n";  // 3. top_of_mid
            // std::cout << "bot_of_mid: " << bot_of_mid << "\n";  // 4. bot_of_mid
            // std::cout << "top_of_leg: " << top_of_leg << "\n";  // 5. top_of_leg
            // std::cout << "bot_of_leg: " << bot_of_leg << "\n";  // 6. bot_of_leg
            // std::cout << "----------------------\n";

            snowman.push_back(x_head); // 7
            snowman.push_back(y_head); // 8

            snowman.push_back(x_neck); // 9
            snowman.push_back(y_neck); // 10

            snowman.push_back(x_chest); // 11
            snowman.push_back(y_chest); // 12

            snowman.push_back(x_lhip); // 13
            snowman.push_back(y_lhip); // 14

            snowman.push_back(x_rhip); // 15
            snowman.push_back(y_rhip); // 16


            snowman.push_back(x_lknee); // 17
            snowman.push_back(y_lknee); // 18

            snowman.push_back(x_rknee); // 19
            snowman.push_back(y_rknee); // 20

            snowman.push_back(x_lankle); // 21
            snowman.push_back(y_lankle); // 22

            snowman.push_back(x_rankle); // 23
            snowman.push_back(y_rankle); // 24


            snowmen.push_back(snowman);
        }

        // ----- CONVERT POSEKEYPOINTS ARRAY TO STD::VECTOR ----- //
        // REORDER OPENPOSE 2D POSES
        std::vector<std::vector<std::vector<float>>> joints_2d_orig;
        for(auto pose = 0; pose < poseKeypoints.getSize(0); pose++)
        {
            std::vector<std::vector<float>> joints;
            for(auto bodyPart = 0; bodyPart < poseKeypoints.getSize(1); bodyPart++)
            {
                std::vector<float> joint;
                joint.push_back(poseKeypoints[{pose, bodyPart, 0}]);
                joint.push_back(poseKeypoints[{pose, bodyPart, 1}]);
                joint.push_back(poseKeypoints[{pose, bodyPart, 2}]);
                joints.push_back(joint);
            }
            joints_2d_orig.push_back(joints);
        }

        // ----------SORT SNOWMEN FROM CLOSEST (BIGGEST RADIUS)---------- //
        float lowest_snowman = -1;
        std::vector<std::vector<float>> snowmen_sorted;
        // REORDER JOINTS_3D POSES
        std::vector<std::vector<std::vector<float>>> joints_3d;
        std::vector<std::vector<std::vector<float>>> datumsPtr_joints_3d = datumsPtr->joints_3d_root_relative;
        std::vector<std::vector<std::vector<float>>> joints_2d;
        for(auto sm = 0; sm < snowmen.size(); sm++)
        {
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

            cv::Point left_top((int)cen_x - rad_bot, (int)top_of_head);
            cv::Point right_top((int)cen_x + rad_bot, (int)top_of_head);
            
            // cv::line(currPose_output, mid_bot, right_bot, cv::Scalar(255,0,0), 1, 8, 0);
            // cv::line(currPose_output, mid_bot, left_bot, cv::Scalar(255,0,0), 1, 8, 0);

            // cv::line(currPose_output, left_bot, left_top, cv::Scalar(255,0,0), 1, 8, 0);
            // cv::line(currPose_output, right_bot, right_top, cv::Scalar(255,0,0), 1, 8, 0);

            // cv::line(currPose_output, left_top, right_top, cv::Scalar(255,0,0), 1, 8, 0);

            cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);
            // cv::line(currPose_output, image_center, left_bot, cv::Scalar(0,0,0), 1, 8, 0);
            
            std::vector<float> snowman_temp = snowmen.at(sm);
            snowman_temp.push_back(left_bot.x); // 25, 7. left_bot.x
            snowman_temp.push_back(left_bot.y); // 26, 8. left_bot.y

            // --> Sort snowman from floor level
            if(bot_of_leg > lowest_snowman)
            {
                snowmen_sorted.insert(snowmen_sorted.begin(), snowman_temp);
                joints_3d.insert(joints_3d.begin(), datumsPtr_joints_3d.at(sm));
                joints_2d.insert(joints_2d.begin(), joints_2d_orig.at(sm));

                lowest_snowman = bot_of_leg;
            }
            else 
            {
                // std::cout << "--------------ELSE FIND A PLACE TO INSERT CURRENT SNOWMAN-----------------\n";
                int s = 0;
                while(true)
                {
                    if(s >= snowmen_sorted.size()) break;
                    if(bot_of_leg > snowmen_sorted.at(s).at(6)) break;
                    s++;
                }

                snowmen_sorted.insert(snowmen_sorted.begin() + s, snowman_temp);
                joints_3d.insert(joints_3d.begin() + s, datumsPtr_joints_3d.at(sm));
                joints_2d.insert(joints_2d.begin() + s, joints_2d_orig.at(sm));
            }

        }

        datumsPtr->joints_3d_root_relative.clear();
        // datumsPtr->joints_3d_root_relative = joints_3d;

        // for(auto i = 0; i < snowmen_sorted.size(); i++)
        // {
        //     std::cout << "check full snowman: " << i << ", full? " << checkFullSnowman(snowmen_sorted.at(i)) << "\n";
        // }


        // std::cout << "----------------------->floorLevelPt: " << floorLevelPt << "\n";

        std::vector<float> poses_relations; // store relative depths, may extend to storing relatiive heights

        // ----- IF NO PREV TO COMPARE, INITIALISE NEW SNOWMAN ----- //
        std::vector<std::vector<float>> prevSnowmen = datumsPtr->prevSnowmen;
        if(prevSnowmen.empty())
        {
            // std::cout << "====> PREV SNOWMEN IS EMPTY <====\n";
            // std::cout << "------->BEFORE into infer_depth:  " << "\n";
            // for(auto s = 0; s < snowmen_sorted.size(); s++)
            // {
            //     std::cout << "snowman s: " << s << " has size: " << snowmen_sorted.at(s).size() << "\n";
            // }

            cv::Point floorLevelPt = getFloorLevelPt(snowmen_sorted.at(0));
            datumsPtr->floorLevelPt = floorLevelPt;
            cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);
            float height = heightInit(snowmen_sorted.at(0), image_center, floorLevelPt);
            datumsPtr->heightInitial = height;
            std::cout << "502: height: " << height << "\n";
            std::cout << "502: datumsPtr->heightInitial: " << datumsPtr->heightInitial << "\n";

            // std::cout << "-> floorLevelPt.x: " << floorLevelPt.x << ", floorLevelPt.y: " << floorLevelPt.y << "\n";

            std::vector<int> toDelete;
            for(auto s = 0; s < snowmen_sorted.size(); s++)
            {

                if(!checkFullSnowman(snowmen_sorted.at(s))) 
                {
                    // INCOMPLETE POSE AND NOT MATCHED, THEREFORE, CANNOT INITIALISE
                        // REMOVE FROM SNOWMEN LIST
                    // std::cout << "cannot make a snowman here\n";
                    toDelete.push_back(s);
                    continue;
                }

                // cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);
                // float infer_depth = inferDepth(snowmen_sorted.at(s).at(7), snowmen_sorted.at(s).at(8), snowmen_sorted.at(s+1).at(7),
                //                 snowmen_sorted.at(s+1).at(8), image_center);

                float infer_depth = inferDepth(floorLevelPt.x, floorLevelPt.y, snowmen_sorted.at(s).at(25), snowmen_sorted.at(s).at(26), image_center);
                // std::cout << "s: " << s << "\n";
                // std::cout << "-> snowman(25): " << snowmen_sorted.at(s).at(25) << ", " << snowmen_sorted.at(s).at(26) << "\n";
                // std::cout << "-->infer_depth: " << infer_depth << "\n";
                snowmen_sorted.at(s).push_back(infer_depth); // 27
                // std::cout << "snowmen_sorted.at(s).at(27): " << snowmen_sorted.at(s).at(27) << "\n";
                // datumsPtr->floorLevels.push_back(infer_depth);
                float heightFound = heightInit(snowmen_sorted.at(s), image_center, floorLevelPt);
                snowmen_sorted.at(s).push_back(heightFound/height); // 28
                // std::vector<cv::Point> heightPts = heightInit(snowmen_sorted.at(s), image_center, floorLevelPt);
                // cv::line(currPose_output, image_center, heightPts.at(2), cv::Scalar(255,0,0), 1, 8, 0);
                // cv::line(currPose_output, image_center, heightPts.at(3), cv::Scalar(255,0,0), 1, 8, 0);
                // cv::line(currPose_output, heightPts.at(2), heightPts.at(3), cv::Scalar(0,0,255), 1, 8, 0);

            }

            // REMOVE ANY INVALID SNOWMAN
            // std::cout << "toDelete size(): " << toDelete.size() << "\n";
            // int m = 0;
            for(int d = toDelete.size() - 1; d >= 0; d--)
            {
                // if(m == 2) break;
                // std::cout << "d: " << d << "\n";
                // m++;

                // std::cout << "toDelete: " << toDelete.at(d) << ", at d: " << d << "\n";
                snowmen_sorted.erase(snowmen_sorted.begin() + (toDelete.at(d)));
                joints_3d.erase(joints_3d.begin() + (toDelete.at(d)));
            }

            // std::cout << "------->AFTER into infer_depth:  " << "\n";
            // for(auto s = 0; s < snowmen_sorted.size(); s++)
            // {
            //     std::cout << "snowman s: " << s << " has size: " << snowmen_sorted.at(s).size() << "\n";
            // }

            // ----FUTURE: FIX IMPERFECT SNOWMEN ON FIRST FRAME TO REDUCE CHANCE OF IMPERFECT DETECTION FROM OPENPOSE
        }
        else
        { 

            // std::cout << "====> PREV SNOWMEN IS not EMPTY <====\n";
            // std::cout << "------->BEFORE into infer_depth:  " << "\n";
            // for(auto s = 0; s < snowmen_sorted.size(); s++)
            // {
            //     std::cout << "snowman s: " << s << " has size: " << snowmen_sorted.at(s).size() << "\n";
            // }


            // ---------- STAGE 1.1: GET STAGE 1 SCORE, COMPARE SNOWMEN SIZES TO GET SCORE OF DIFF IN RADII ---------- //
            std::vector<float> score_stage1;
            for(auto i = 0; i < snowmen_sorted.size(); i++)
            {
                // ----> COMPARE WITH ALL PREV SNOWMEN TO ALL CURR SNOWMEN
                for(auto p = 0; p < prevSnowmen.size(); p++)
                {
                    float score = compareSnowmenSize(snowmen_sorted.at(i), prevSnowmen.at(p));
                    // std::cout << "->score: " << abs((score)) << "\n";
                    score_stage1.push_back(score == 0 ? 0 : abs(score)); // not considering direction
                }

            }

            // ---------------- STAGE 1.2: MATCH SNOWMAN, FIND LOWEST SCORE --------------- //
            std::vector<int> scstage1_indices;
            std::vector<int> scstage1_scores;
            auto score_index = 0;
            for(auto c = 0; c < snowmen_sorted.size(); c++)
            {
                // std::cout << "curr snowmen: " << c << "\n";

                std::vector<float> score_stage1_temp;
                for(auto p = 0; p < prevSnowmen.size(); p++)
                {
                    // std::cout << "->sc: " << score_index << ", score: " << score_stage1.at(score_index) << "\n";
                    score_stage1_temp.push_back(score_stage1.at(score_index));
                    score_index++;
                }
                // std::cout << "-----------check 1---------\n";
                float score_min_temp = vNectFindMin(score_stage1_temp);
                // std::cout << "-----------check 2---------\n";
                int score_min_index = getVecIndex(score_stage1_temp, score_min_temp, 0); // SHOULD NOT FAIL TO FIND (-1)
                // std::cout << "..> --> score_stage1_temp.size(): " << score_stage1_temp.size() << "\n";
                // std::cout << "      -->> score_min_index: " << score_min_index << "\n";
                scstage1_indices.push_back(score_min_index);
                scstage1_scores.push_back(score_min_temp);
                // std::cout << "-----------check 3---------\n";
            }
            // std::cout << "---------\n";

            // std::cout << "---> prevSnowmen.at(scstage1_indices.at(i)): " << prevSnowmen.at(0).at(0) << "\n";  
            
            // ---------------- STAGE 1.3: CHECK ALL CURR IS MATCHED (COLLECT UNMATCHED) --------------- //
            // ---------------- STAGE 1.4: CHECK ALL PREV IS MATCHED (COLLECT UNMATCHED) --------------- //

            // ------------ STAGE 1.5.1: UNMATCHED CURR AND PREV EXIST -> COMPARE UNMATCHED CURR AND PREV ------------- //
            // ------------ STAGE 1.5.2: UNMATCHED PREV EXIST ONLY -> COMPARE UNMATCHED PREV WITH UNMATCHED DETECTED POSEKEYPOINTS --------- //
                // IF NO MORE DETECTED, IGNORE (SKIP EVEN IF THERE EXIST, BUT OP DOES NOT DETECT IN THE FIRST PLACE) //

            // ------------ STAGE 1.5.3: UNMATCHED CURR EXIST ONLY -> GO INITIALISE NEW SNOWMAN ------------ //

            // for(auto i = 0; i < scstage1_scores.size(); i++) std::cout << i << ". --> scstage1_scores: " << scstage1_scores.at(i) << "\n";

            
            // ----> CHECK FOR PREV COMPETE
            // std::vector<int> tempv = getOccIndices(scstage1_indices, 1);
            // for(auto i = 0; i < tempv.size(); i++)
            // {
            //     std::cout << "~getOccIndices i: " << i << ", results: " << tempv.at(i) << "\n";
            // }

            // std::cout << "----> BEFORE FOUND UNMATCHED PREV AND CURR\n";
            // for(auto i = 0; i < scstage1_indices.size(); i++)
            // {
            //     std::cout << "i: " << i << ", : " << scstage1_indices.at(i) << "\n";
            //     std::cout << "  i: " << i << ", : " << scstage1_scores.at(i) << "\n";
            // }
            // (snowmen_sorted) currSnowmen stores all complete and incomplete snowmen        e.g. {0 vec<float>,1 vec<float>, 2 vec<float>,3,4,....}
            // prevSnowmen stores all complete and incomplete snowmen        e.g. {0 vec<float>,1 vec<float>, 2 vec<float>,3,4,....}
            // scstage1_indices stores all local minimum index for all currs e.g. {9,3,1,2...., -1,....}
            // scstage1_scores stores all local minimum scores for all currs e.g. {0.3,0.4,0.2,0.9....,...}
            cv::Point floorLevelPt = datumsPtr->floorLevelPt;
            float height = datumsPtr->heightInitial;
            std::cout << "645: height: " << height << "\n";
            
            std::vector<float> prevSnow_unmatched;
            std::vector<float> currSnow_unmatched;
            for(auto p = 0; p < prevSnowmen.size(); p++)
            {
                // elements in scstage1_indices will not have value greater then prevSnowmen size, because they are local index/scores
                // std::cout << "--> getting occurence of " << p << "=>" << std::count(scstage1_indices.begin(), scstage1_indices.end(), (int) p) << "\n";
                
                // std::cout << "getVecIndex 0,0: " << getVecIndex(scstage1_indices, 0, 0) << "\n";
                // std::cout << "getVecIndex 1,0: " << getVecIndex(scstage1_indices, 1, 0) << "\n";
                // std::cout << "getVecIndex 0,2: " << getVecIndex(scstage1_indices, 0, 2) << "\n"; 
                // std::cout << "getVecIndex 1,1: " << getVecIndex(scstage1_indices, 1, 1) << "\n";

                // IF OCCURENCE == 0 => PREV UNMATCHED
                // IF OCCURENCE == 1 => PERFECT MATCH; SKIP
                // IF OCCURENCE > 1 => COMPETE FOR PREV; COMPARE SCORE AND REMATCH OR REINITIALISE THE UNMATCHED ONE

                if(std::count(scstage1_indices.begin(), scstage1_indices.end(), (int) p) == 1)
                {
                    // NEED TO UPDATE DEPTH
                    // std::cout << "---------> PERFECT MATCH <------ AT prev p = " << p << "\n";
                    std::vector<int> tempv = getOccIndices(scstage1_indices, (int) p); // FINDING INDEX OCCURENCE in scstage1_indices
                    // tempv here will only contain exactly one index which is the curr index
                     // CHECK IF THE MATCH IS APPROPRIATE, NOT JUST NUMERICALLY MATCHED

                    // FOR AN APPROPRIATE MATCH, UPDATE THE SNOWMAN
                    // if(checkFullSnowman(snowmen_sorted.at(tempv.at(0))))
                    // {
                        // std::cout << "successful PERFECT Match\n";
                        snowmen_sorted.at(tempv.at(0)) = updateSnowman(prevSnowmen.at(p), snowmen_sorted.at(tempv.at(0)), floorLevelPt);
                        // std::cout << "UPADTE SNOWMAN 1: SIZE: " << snowmen_sorted.at(tempv.at(0)).size() << "\n";
                    // }
                    // else
                    // {
                    //     std::cout << "not PERFECT Match\n";
                    //     // SET BOTH CURR AND PREV TO UNMATCH
                    //     currSnow_unmatched.push_back(tempv.at(0)); // UNMATCHED CURR,, SAVING INDEX OF THE CURR
                    //     prevSnow_unmatched.push_back(p);
                    // }
                    continue;
                }
                else if(std::count(scstage1_indices.begin(), scstage1_indices.end(), (int) p) == 0)
                {
                    // --> UNMATCHED PREV,, SAVING INDEX OF THE PREV
                    // std::cout << "---------> UNMATCHED PREV <------ AT prev p = " << p << "\n";
                    prevSnow_unmatched.push_back(p);
                }
                else if(std::count(scstage1_indices.begin(), scstage1_indices.end(), (int) p) > 1)
                {   
                    // std::cout << "---------> COMPETING MATCH <------ AT prev p = " << p << "\n";
                    // 1. IF N CURR COMPETE FOR 1 PREV
                    // 2. ONLY ONE CURR GETS THE PREV AND OTHERS SET TO UNMATCHED

                    // 3. GET ALL CURR (INDEX) COMPETING FOR THE PREV P
                    std::vector<int> tempv = getOccIndices(scstage1_indices, (int) p); // FINDING INDEX OCCURENCE in scstage1_indices
                    // std::cout << "--------> getting all index occurence of local minimum score, prev at: " << p << "\n";
                    
                        // 3.1. tempv stored all competing curr indices e.g. {2,5,9}

                    float localMinScore = 10000;
                    int globalMinIndex = -1;

                    // 4. GET THE SCORES OF THOSE CURRS AGAINST THE PREV
                    std::vector<float> tempsc;
                    for(auto t = 0; t < tempv.size(); t++)
                    {
                        tempsc.push_back(scstage1_scores.at(tempv.at(t)));
                    }

                        // 4.1. tempsc stores all competing curr scores e.g. {2->0.3, 5->0.9, 9->0.2}

                    // for(auto t= 0; t < tempsc.size(); t++)
                    // {
                    //     std::cout << "t: " << t << ", tempsc: " << tempsc.at(t) << "\n";
                    // }

                    // 5. FIND THE LOCAL MINIMUM SCORE OF OUT ALL COMPETING CURRS
                    for(auto t = 0; t < tempv.size(); t++)
                    {   
                        // e.g. scstage_1 = [0,0], then tempv.at(0) and at(1) = [0,1] <= this [0,1] is the indices of scstage_1 that hold "0"
                        // std::cout << "--------> tempv: " << t << ", results index at: " << tempv.at(t) << ", score: " << scstage1_scores.at(tempv.at(t)) << "\n";
                        if(localMinScore > scstage1_scores.at(tempv.at(t)))
                        {
                            localMinScore = scstage1_scores.at(tempv.at(t));
                            globalMinIndex = tempv.at(t);
                        }
                    }

                    // std::cout << "localMinScore: " << localMinScore << "\n";
                    // std::cout << "globalMinIndex: " << globalMinIndex << "\n";

                    // 6. SET COMPETING CURRS TO UNMATCHED VALUE IN scstage1_indices and scstage1_scores
                        // EXCEPT FOR THE CURR WITH LOWEST LOCAL MINIMUM SCORE

                    for(auto t = 0; t < tempv.size(); t++)
                    {
                        // std::cout << "t: " << t << ", tempv at t: " << tempv.at(t) << "\n";
                        // std::cout << "  scstage1_indices.at(tempv.at(t)): " << scstage1_indices.at(tempv.at(t)) << "\n";
                        if(tempv.at(t) != globalMinIndex)
                        {
                            // std::cout << "?????????? only once right ?????\n";
                            // scstage1_indices.at(x) where x is     
                                scstage1_indices.at(tempv.at(t)) = -1;
                                scstage1_scores.at(tempv.at(t)) = -1;
                                currSnow_unmatched.push_back(tempv.at(t)); // UNMATCHED CURR,, SAVING INDEX OF THE CURR
                        }
                    }

                    snowmen_sorted.at(globalMinIndex) = updateSnowman(prevSnowmen.at(p), snowmen_sorted.at(globalMinIndex), floorLevelPt);
                    // std::cout << "UPADTE SNOWMAN 2: SIZE: " << snowmen_sorted.at(globalMinIndex).size() << "\n";
                    // std::cout << "prevSnowman to update after compete\n";
                    // std::cout << "at(0): " << prevSnowmen.at(p).at(0) << "\n";
                    // std::cout << "snowmen_sorted at(globalMinIndex) at(0): " << snowmen_sorted.at(globalMinIndex).at(0) << "\n";


                }
            }


            // for(int i = 0; i < snowmen_sorted.size(); i++)
            // {
            //     std::cout << "snowmen_sorted i: " << i << ", at(0): " << snowmen_sorted.at(i).at(0) << "\n";
            // }


            // std::cout << "----> AFTER FOUND UNMATCHED PREV AND CURR\n";
            // for(auto i = 0; i < scstage1_indices.size(); i++)
            // {
            //     std::cout << "i: " << i << ", : " << scstage1_indices.at(i) << "\n";
            //     std::cout << "  i: " << i << ", : " << scstage1_scores.at(i) << "\n";
            // }
            // // ----> CHECK FOR UNMATCHED CURR AND PREV SNOWMEN
            // std::vector<int> currSnow_unmatched;
            // std::vector<int> prevSnow_matched;
            // for(auto c = 0; c < scstage1_indices.size(); c++)
            // {
            //     std::cout << "c: " << c << ", scstage1_indices.at(c): " << scstage1_indices.at(c) << "\n";
            //     // IF -1, ADD THE CURR UNMATCHED SNOWMAN INDEX TO LIST
            //     if(scstage1_indices.at(c) == -1)
            //     {
            //         currSnow_unmatched.push_back((int) c);
            //         // continue;
            //     }

            //     // IF !-1, ADD THE INDEX TO LIST TO FIND UNMATCHED PREV SNOWMAN
            //     else
            //     {
            //         prevSnow_matched.push_back((int) c);
            //     }
            // }

            // std::vector<int> prevCheck_index;
            // for(auto i = 0; i < prevSnowmen.size(); i++) prevCheck_index.push_back(i);

            // std::vector<int> prevSnow_unmatched;

            // std::vector<int>::iterator it;
            // for(auto i = 0; i < prevSnow_matched.size(); i++)
            // {
            //     it = std::find(prevCheck_index.begin(), prevCheck_index.end(), prevSnow_matched.at(i));
            //     if(it == prevCheck_index.end())
            //     {
            //         prevSnow_unmatched.push_back((int) i);
            //     }
            // }

            // if(prevSnow_unmatched.empty())
            // {
            //     std::cout << "-----------------------------------------------------------------prevSnow_unmatched is empty!!!! Yay!\n";
            // }

            // if(currSnow_unmatched.empty())
            // {
            //     std::cout << "-----------------------------------------------------------------currSnow_unmatched is empty!!!! Yay!\n";
            // }

            // std::cout << "--------------\n";

            // while(true)
            // {

            //     std::cout << "prevSnow_unmatched size(): " << prevSnow_unmatched.size() << "\n";
            //     std::cout << "currSnow_unmatched size(): " << currSnow_unmatched.size() << "\n";
            //     break;
            // }

            // REMATCH PREV AND CURR
            // float threshold_rematch = 1000;
            std::vector<int> curr_rematched;
            for(auto i = 0; i < prevSnowmen.size(); i++)
            {
                // IF PREV IS ALREADY MATCHED WITH A CURR (IF NOT IN THE UNMATCHED LIST), SKIP
                if(std::find(prevSnow_unmatched.begin(), prevSnow_unmatched.end(), (int) i) == prevSnow_unmatched.end()) continue;
                // std::cout << "prevSnow_unmatched at i " << i << " = " << prevSnow_unmatched.at(i) << "\n";

                std::vector<float> scores_rematch_temp;
                std::vector<int> indices_rematch_temp;
                // std::cout << "prevSnowmen to rematch is at index: " << i << "\n";
                for(auto j = 0; j < snowmen_sorted.size(); j++)
                {

                    if(std::find(currSnow_unmatched.begin(), currSnow_unmatched.end(), (int) j) == currSnow_unmatched.end()) continue;
                    if(std::find(curr_rematched.begin(), curr_rematched.end(), (int) j) != curr_rematched.end()) continue;
                    // std::cout << "----->snowmen_sorted to rematch is at index: " << j << "\n";

                    // REMATCH METHOD
                    float score = reMatchSnowmen(prevSnowmen.at(i), snowmen_sorted.at(j));
                    // if(score > threshold_rematch) continue;
                    // std::cout << "score: " << score << "\n";
                    if(score == -1) continue;
                    // ONLY KEEP VALID SCORES AND CURR CANDIDATES
                    scores_rematch_temp.push_back(score);
                    indices_rematch_temp.push_back(j);

                }

                // 3. GET THE LOWEST MINIMUM CURR AND MATCH TO PREV (MATCH = MODIFY THE CURR TO ADJUST TO PREV)
                if(scores_rematch_temp.empty()) continue; // IF NO REALLY NO MATCH (POSE MAY LEAVE FRAME)

                float score_min_temp = vNectFindMin(scores_rematch_temp);
                int score_min_index = getVecIndex(scores_rematch_temp, score_min_temp, 0); // SHOULD NOT FAIL TO FIND (-1)
                score_min_index = indices_rematch_temp.at(score_min_index);

                // std::cout << "score_min_temp: " << score_min_temp << ", score_min_index: " << score_min_index << "\n";
                // MATCH I.E. MODIFY THE CURR AT THIS INDEX !!
                std::vector<float> sm_temp = reAdjustSnowman(prevSnowmen.at(i), snowmen_sorted.at(score_min_index));
                snowmen_sorted.at(score_min_index) = sm_temp;

                // std::cout << "sm_temp size(): " << sm_temp.size() << "\n";

                // NO NEED TO REMOVE MATCHED PREV SINCE WILL NOT COME BACK
                // NEED TO REMOVE MATCHED AND CURR from UNMATCHED LIST
                curr_rematched.push_back(score_min_index);
            }

            // std::cout << "-------> 2 BEFORE into infer_depth:  " << "\n";
            // for(auto s = 0; s < snowmen_sorted.size(); s++)
            // {
            //     std::cout << "snowman s: " << s << " has size: " << snowmen_sorted.at(s).size() << "\n";
            // }

            // std::cout << "---> display currSnow_unmatched <---\n";
            // for(auto i = 0; i < currSnow_unmatched.size(); i++)
            // {
            //     std::cout << "i = " << currSnow_unmatched.at(i) << "\n";
            // }

            // std::cout << "---> display curr_rematched <---\n";
            // for(auto i = 0; i < curr_rematched.size(); i++)
            // {
            //     std::cout << "i = " << curr_rematched.at(i) << "\n";
            // }


            std::vector<int> toDelete;
            // cv::Point floorLevelPt = datumsPtr->floorLevelPt;
            for(auto i = 0; i < snowmen_sorted.size(); i++)
            {   
                // std::cout << "---> reinitialise snowman\n";
                if(std::find(currSnow_unmatched.begin(), currSnow_unmatched.end(), (int) i) == currSnow_unmatched.end()) continue;
                if(std::find(curr_rematched.begin(), curr_rematched.end(), (int) i) != curr_rematched.end()) continue;
                // std::cout << "pass i = " << i << "\n";

                if(!checkFullSnowman(snowmen_sorted.at(i))) 
                {
                    // INCOMPLETE POSE AND NOT MATCHED, THEREFORE, CANNOT INITIALISE
                        // REMOVE FROM SNOWMEN LIST
                    // std::cout << "cannot make a snowman here\n";
                    toDelete.push_back(i);
                    continue;
                }
                cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);

                float infer_depth = inferDepth(floorLevelPt.x, floorLevelPt.y, snowmen_sorted.at(i).at(25), snowmen_sorted.at(i).at(26), image_center);
                snowmen_sorted.at(i).push_back(infer_depth); // 27
                // datumsPtr->floorLevels.push_back(infer_depth);

                float heightFound = heightInit(snowmen_sorted.at(i), image_center, floorLevelPt);
                std::cout << "heightFound: " << heightFound << "\n";
                std::cout << "  height: " << height << "\n";
                snowmen_sorted.at(i).push_back(heightFound/height); // 28
            }

            // REMOVE ANY INVALID SNOWMAN
            // std::cout << "toDelete size(): " << toDelete.size() << "\n";
            // int m = 0;
            for(int d = toDelete.size() - 1; d >= 0; d--)
            {
                // if(m == 2) break;
                // std::cout << "d: " << d << "\n";
                // m++;

                // std::cout << "toDelete: " << toDelete.at(d) << ", at d: " << d << "\n";
                snowmen_sorted.erase(snowmen_sorted.begin() + (toDelete.at(d)));
                joints_3d.erase(joints_3d.begin() + (toDelete.at(d)));
            }

            // std::cout << "--DELETED A SNOWMAN\n";
            // std::cout << "snowmen_sorted.size(): " << snowmen_sorted.size() << "\n";
            // std::cout << "snowmen_sorted.at(0).size(): " << snowmen_sorted.at(0).size() << "\n";

            // for(auto i = 0; i < snowmen_sorted.size(); i++)
            // {
            //     std::cout << "~~~~> snowman size(): " << snowmen_sorted.at(i).size() << "\n";
            // }
            // // SET NEW PREV SNOWMEN
            // if(!prevSnowmen.empty())
            // {
            //     for(auto i = 0; i < scstage1_indices.size(); i++)
            //     {
            //         std::cout << "i: " << i << ", " << scstage1_indices.at(i) << "\n";
            //         std::cout << "  global index: " << (i * (prevSnowmen.size())) + scstage1_indices.at(i) << "\n";

            //         // std::cout << "---> prevSnowmen.at(scstage1_indices.at(i)): " << prevSnowmen.at(0).at(0) << "\n";
            //         // std::cout << "---> prevSnowmen size: " << prevSnowmen.size() << "\n";
            //         // std::cout << "---> scstage1_indices size: " << scstage1_indices.size() << "\n";
            //         // std::cout << "---> scstage1_indices.at(i): " << scstage1_indices.at(i) << "\n";


            //         int cen_x = prevSnowmen.at((int)scstage1_indices.at(i)).at(0);
            //         // std::cout << "----->check 1\n";
            //         float top_of_head = prevSnowmen.at(scstage1_indices.at(i)).at(1);
            //         float bot_of_head = prevSnowmen.at(scstage1_indices.at(i)).at(2);
            //         float top_of_mid = prevSnowmen.at(scstage1_indices.at(i)).at(3);
            //         float bot_of_mid = prevSnowmen.at(scstage1_indices.at(i)).at(4);
            //         float top_of_leg = prevSnowmen.at(scstage1_indices.at(i)).at(5);
            //         float bot_of_leg = prevSnowmen.at(scstage1_indices.at(i)).at(6);
            //         float x_left_bot_a = prevSnowmen.at(scstage1_indices.at(i)).at(7);
            //         float y_left_bot_a = prevSnowmen.at(scstage1_indices.at(i)).at(8);

            //         // std::cout << "----->check 2\n";
            //         float rad_top = bot_of_head - (top_of_head + bot_of_head) / 2;
            //         float rad_mid = bot_of_mid - (top_of_mid + bot_of_mid) / 2;
            //         float rad_bot = bot_of_leg - (top_of_leg + bot_of_leg) / 2;
                    
            //         // std::cout << "----->check 3\n";
            //         // DRAW SNOWMAN ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            //         // cv::circle(currPose, cv::Point(cen_x, (int) (top_of_head + bot_of_head)/2), rad_top, cv::Scalar(0,255,0), -1, 8);
            //         // cv::circle(currPose, cv::Point(cen_x, (int) (top_of_mid + bot_of_mid)/2), rad_mid, cv::Scalar(0,255,0), -1, 8);
            //         // cv::circle(currPose, cv::Point(cen_x, (int) (top_of_leg + bot_of_leg)/2), rad_bot, cv::Scalar(0,255,0), -1, 8);

            //         // // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
            //         // // cv::circle(currPose, cv::Point(cen_x, (int)top_of_leg), 5, cv::Scalar(0,255,255), -1, 8);
            //         // // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
            //         // // cv::circle(currPose, cv::Point(cen_x, (int)top_of_mid), 5, cv::Scalar(0,255,255), -1, 8);
            //         // // cv::circle(currPose, cv::Point(cen_x, (int)bot_of_head), 5, cv::Scalar(0,255,255), -1, 8);
            //         // // cv::circle(currPose, cv::Point(cen_x, (int)top_of_head), 5, cv::Scalar(0,255,255), -1, 8);

            //         // cv::addWeighted(currPose, opacity, currPose_output, 1 - opacity, 0, currPose_output); 
            //         ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            //     }
            // }

            // std::cout << "||||||||||||||||||||||||||||||||||||||||||||||||\n";
            // }

            // auto size_orient;
            // if(datumsPtr->prevOrientation.size() > datumsPtr->orientation.size()) size_orient = datumsPtr->orientation.size();
            // else if(datumsPtr->prevOrientation.size() <= datumsPtr->orientation.size()) size_orient = datumsPtr->prevOrientation.size();

            // for(auto c = 0; c < datumsPtr->orientation.size(); c++)
            // {
            //     if(datumsPtr->orientation.at(c) == 0) std::cout << "orientation is zero!!\n";
            //     for(auto p = 0; p < datumsPtr->prevOrientation.size(); p++)
            //     {
            //         // std::cout << "i: " << i << "prevOri: " << datumsPtr->prevOrientation.at(i) << "\n";
            //         if(datumsPtr->prevOrientation.at(p) == 0) std::cout << "prevOrientation is zero!!\n";
            //     }

            // }
            // std::cout << "-----------<<<<<<<<<END OF UPDATE PREV>>>>>>>>>-------------\n";
        }

                // --------- SAVE SNOWMEN TO DATUM --------- //
        for(auto cs = 0; cs < snowmen_sorted.size(); cs++)
        {
            // std::cout << "CHECK SNOWMAN SIZE: " << snowmen_sorted.at(cs).size() << "\n";
            // std::cout << "at(28): " << snowmen_sorted.at(cs).at(28) << "\n";
            datumsPtr->snowmen.push_back(snowmen_sorted.at(cs));
            // datumsPtr->orientation.push_back(joints_2d.at(i).at(14).at(0)); // x chest
            // datumsPtr->orientation.push_back(joints_2d.at(i).at(14).at(1)); // y chest
            // std::cout << "---->snowmen_sorted: " << cs << "------------------\n";
            int cen_x = snowmen_sorted.at(cs).at(0);
            float top_of_head = snowmen_sorted.at(cs).at(1);
            float bot_of_head = snowmen_sorted.at(cs).at(2);
            float top_of_mid = snowmen_sorted.at(cs).at(3);
            float bot_of_mid = snowmen_sorted.at(cs).at(4);
            float top_of_leg = snowmen_sorted.at(cs).at(5);
            float bot_of_leg = snowmen_sorted.at(cs).at(6);
            // std::cout << "bot_of_leg: " << bot_of_leg << "\n";
            // std::cout << "top_of_leg: " << top_of_leg << "\n";
            // std::cout << "bot_of_mid: " << bot_of_mid << "\n";
            // std::cout << "top_of_mid: " << top_of_mid << "\n";
            // std::cout << "bot_of_head: " << bot_of_head << "\n";
            // std::cout << "top_of_head: " << top_of_head << "\n";
            // std::cout << "cen_x: " << cen_x << "\n";
            float scale = snowmen_sorted.at(cs).at(28);
            std::vector<std::vector<float>> joi3d = joints_3d.at(cs);
            float left_toe_offset = joi3d.at(17).at(1) - (joi3d.at(17).at(1) * scale);
            // std::cout << "scale: " << scale << "\n";
            // std::cout << "joi3d.at(17).at(1): " << joi3d.at(17).at(1) << "\n";
            // std::cout << "left_toe_offset: " << left_toe_offset << "\n";
            for(int j = 0; j < joi3d.size(); j++)
            {
                // if(j % 3 == 2) continue;
                // std::cout << "joi3d.at(j).at(0): " << joi3d.at(j).at(0) << "\n";
                // std::cout << "joi3d.at(j).at(1): " << joi3d.at(j).at(1) << "\n";
                
                float xval = joi3d.at(j).at(0) * scale;
                // std::cout << "xval 1: " << xval << "\n";
                xval = xval - left_toe_offset;
                // std::cout << "xval 2: " << xval << "\n";
                joi3d.at(j).at(0) = xval;

                float yval = joi3d.at(j).at(1) * scale;
                // std::cout << "yval 1: " << yval << "\n";
                yval = yval - left_toe_offset;
                // std::cout << "yval 2: " << yval << "\n";
                joi3d.at(j).at(1) = yval;
            }

            joints_3d.at(cs).clear();
            joints_3d.at(cs) = joi3d;
        }

        datumsPtr->joints_3d_root_relative = joints_3d;


        // ----------------------------------- GET DISTANCES BETWEEN EACH PAIR POSE-------------------------------- //
        for (int person = 0 ; person < joints_2d.size(); person++)
        {
            cv::Point image_center((int) currPose_output.cols/2, (int) currPose_output.rows/2);

            // float neckDiff = (poseKeypoints[{person+1, 1, 0}] - poseKeypoints[{person, 1, 0}]);
            float neckDiff = image_center.x - joints_2d.at(person).at(1).at(0);
            // float chestDiff = (poseKeypoints[{person+1, 14, 0}] - poseKeypoints[{person, 14, 0}]);
            float chestDiff = image_center.x - joints_2d.at(person).at(14).at(0);

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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// // for now, kind of assuming every pose has the same height or whatever height given by VNect
		// for(int i = 0; i < (int)poseFloorLevel.size()-1; i++)
		// {
		// 	std::cout << "floor level difference: " << (float) abs(poseFloorLevel.at(i) - poseFloorLevel.at(i+1)) << "\n";
		// 	datumsPtr->floorLevels.push_back((float) abs(poseFloorLevel.at(i) - poseFloorLevel.at(i+1)));
		// }
		// // compare radius
		// // depth depends on poses interdependency positions mainly, not where there are in the original image

		// cv::imshow("currPose_output_internal", currPose_output);
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