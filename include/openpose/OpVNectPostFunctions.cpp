// #include <openpose/flags.hpp>
// #include <openpose/headers.hpp>
#include <string>


#include <ncurses.h>
#include <python2.7/Python.h>

void matplotlibVNect()
{
    PyObject* pInt;
    char filename[] = "drawVNect_single.py";
    FILE* fp;
    Py_Initialize();
    // PyRun_SimpleString("print('Hello World from Embedded Python!!!')");
    fp = fopen(filename, "r");
    PyRun_SimpleFile(fp, filename);    
    Py_Finalize();   
}

void print3Ddepths(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, const std::string pathToWrite, const std::string writeAs)
{
    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            auto& floorLevels = datumsPtr->at(0)->floorLevels;
            auto& snowmen = datumsPtr->at(0)->snowmen;

            // std::string fileName = "floorLevels_" + writeAs;
            std::string fileName = writeAs;
            std::string saveToName = pathToWrite + fileName + ".txt";
            std::ofstream out3djoints(saveToName);
            std::streambuf *coutbuf3djoints = std::cout.rdbuf();
            std::cout.rdbuf(out3djoints.rdbuf());

            for(auto i = 0; i < snowmen.size(); i++)
            {
                std::cout << "POSE_" << std::to_string(i) << " " << snowmen.at(i).at(27) * (1) << "\n";
            }

            std::cout.rdbuf(coutbuf3djoints);
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "ERROR WRITING VNECT\n";
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}

void write3dJointsVNect(const std::shared_ptr<std::vector<std::shared_ptr<op::Datum>>>& datumsPtr, const std::string pathToWrite, const std::string writeAs)
{
    try
    {
        if (datumsPtr != nullptr && !datumsPtr->empty())
        {
            ////////////---- GET JOINTS 3D ROOT RELATIVE ----////////////
            auto& joints3d = datumsPtr->at(0)->joints_3d_root_relative;
            // std::cout << "joints3d size: " << joints3d.size() << "\n";
            // std::cout << "joints3d (int)size: " << (int)joints3d.size() << "\n";
            // for(int i = 0; i < joints3d.size(); i++)
            // {
            //     std::cout << "i: " << i << "\n";
            //     for(int j = 0; j < joints3d.at(i).size(); j++)
            //     {
            //         // std::cout << "j = " << j << ", size: " << joints3d.at(i).at(j).size() << "\n";
            //         for(int k = 0; k < joints3d.at(i).at(j).size(); k++)
            //         {
            //             std::cout << joints3d.at(i).at(j).at(k) << " ";
            //         } 
            //         std::cout << "\n";
            //     }
            // }
            /////////////////////////////////////////////////////////////

            /////////---- GET NECK DIFFS ----////////////////////////////
            auto& neckDiffs = datumsPtr->at(0)->neckDiffs;
            // std::cout << "neckDiffs size: " << neckDiffs.size() << "\n";
            // for(int i = 0; i < neckDiffs.size(); i++)
            // {
            //     std::cout << i << ": " << neckDiffs.at(i) << "\n";
            // }

            ////////---- GET CHEST DIFFS ----////////////////////////////
            auto& chestDiffs = datumsPtr->at(0)->chestDiffs;
            // std::cout << "chestDiffs size: " << chestDiffs.size() << "\n";
            // for(int i = 0; i < chestDiffs.size(); i++)
            // {
            //     std::cout << i << ": " << chestDiffs.at(i) << "\n";
            // }
            ///////----- GET AVG DIFFS FROM NECK AND CHEST -----/////////
            // float avgDiffs[joints3d.size()][joints3d.size()];
            // std::cout << "avgDiffs size: " << sizeof(avgDiffs) << "\n";
            // std::cout << "float size: " << sizeof(float) << "\n";
            // int numPoses = joints3d.size();
            // int numPoseRelations = (numPoses * (numPoses - 1))/2;
            // int relationI = 0;

            // for(int i = 0; i < numPoses; i++)
            // {
            //     for(int j = 0; j < numPoses; j++)
            //     {
            //         avgDiffs[i][j] = neckDiffs.at(relationI) - chestDiffs.at(relationI++);
            //     }
            // }

            // std::cout << "under numPoseRelations\n";
            // for(int i = 0; i < numPoseRelations; i++)
            // {
            //     std::cout << "i: " << i << "\n";
            // }

            std::vector<float> avgDiffs;

            for(int i = 0; i < neckDiffs.size(); i++)
            {
                avgDiffs.push_back((neckDiffs.at(i)+chestDiffs.at(i)) / 2);
            }

            // std::cout << "printing avgDiffs\n";
            // for(int i = 0; i < avgDiffs.size(); i++)
            // {
            //     std::cout << i << ". " << avgDiffs.at(i) << "\n";
            // }
            /////////////////////////////////////////////////////////////

            /////////---- GET FLOOR LEVELS ----//////////////////////////
            auto& floorLevels = datumsPtr->at(0)->floorLevels;
            // std::cout << "floorLevels\n";
            // for(int i = 0; i < floorLevels.size(); i++)
            // {
            //     std::cout << i << ". " << floorLevels.at(i) << "\n";
            // }

            /////////////////////////////////////////////////////////////



                /*---- PRINT VNECT_3D_JOINTS.TXT ----*/
        // "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/3d_joints/"
        // std::string pathToWrite = "/home/mooktj/Desktop/myworkspace/mook-openpose/openpose/outputs/OP_VNECT/outputs-7-5-19/check_depths/";
        // std::string fileName = "joints_3d_" + writeAs;
        std::string fileName = writeAs;
        std::string saveToName = pathToWrite + fileName + ".txt";
        std::ofstream out3djoints(saveToName);
        std::streambuf *coutbuf3djoints = std::cout.rdbuf();
        std::cout.rdbuf(out3djoints.rdbuf());

        // std::cout << "joints3d size: " << joints3d.size() << "\n";
        // std::cout << "joints3d.at(0) size: " << joints3d.at(0).size() << "\n";
        // std::cout << "joints3d.at(0).at(0) size: " << joints3d.at(0).at(0).size() << "\n";

        for(int i = 0; i < joints3d.size(); i++)
        {
            for(int j = 0; j < joints3d.at(i).size(); j++)
            {   
                std::cout << "pose_" << i << " ";
                for(int k = 0; k < joints3d.at(i).at(j).size(); k++)
                {
                    std::cout << joints3d.at(i).at(j).at(k) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }

        std::cout << "POSE_END\n\n";

        for(auto x : avgDiffs)
        {
            std::cout << "avgDiffs " << x << "\n";
        }

        std::cout << "AVGDIFFS_END\n\n";

        // for(auto y : floorLevels)
        // {
        //     std::cout << "floorLevels " << y << "\n";
        // }

        // std::cout << "FLOORLEVELS_END\n\n";

        std::cout.rdbuf(coutbuf3djoints);

        }
    }
    catch (const std::exception& e)
    {
        std::cout << "ERROR WRITING VNECT\n";
        op::error(e.what(), __LINE__, __FUNCTION__, __FILE__);
    }
}