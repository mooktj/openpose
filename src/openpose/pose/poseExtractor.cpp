#include <openpose/pose/poseExtractor.hpp>
#include <iostream>

namespace op
{
    const std::string errorMessage = "Either person identification (`--identification`) must be enabled or"
                                     " `--number_people_max 1` in order to run the person tracker (`--tracking`).";

    PoseExtractor::PoseExtractor(const std::shared_ptr<PoseExtractorNet>& poseExtractorNet,
                                 const std::shared_ptr<KeepTopNPeople>& keepTopNPeople,
                                 const std::shared_ptr<PersonIdExtractor>& personIdExtractor,
                                 const std::shared_ptr<std::vector<std::shared_ptr<PersonTracker>>>& personTrackers,
                                 const int numberPeopleMax, const int tracking) :
        mNumberPeopleMax{numberPeopleMax},
        mTracking{tracking},
        spPoseExtractorNet{poseExtractorNet},
        spKeepTopNPeople{keepTopNPeople},
        spPersonIdExtractor{personIdExtractor},
        spPersonTrackers{personTrackers}
    {
        // std::cout << "poseExtractor:: PoseExtractor(...) constructor\n";
    }

    PoseExtractor::~PoseExtractor()
    {
    }

    void PoseExtractor::initializationOnThread()
    {
        // std::cout << "poseExtractor:: initializationOnThread()\n";
        try
        {
            spPoseExtractorNet->initializationOnThread();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractor::forwardPass(const std::vector<Array<float>>& inputNetData,
                                    const Point<int>& inputDataSize,
                                    const std::vector<double>& scaleInputToNetInputs,
                                    const Array<float>& poseNetOutput,
                                    const long long frameId)
    {
        // std::cout << "poseExtractor:: forwardPass(...)" << "\n";
        try
        {
            if (mTracking < 1 || frameId % (mTracking+1) == 0)
            {
                spPoseExtractorNet->forwardPass(inputNetData, inputDataSize, scaleInputToNetInputs, poseNetOutput);
                // std::cout << "---->spPoseExtractorNet->forwardPass" << "\n";
            }
            else
            {
                spPoseExtractorNet->clear();
                // std::cout << "---->spPoseExtractorNet->clear()" << "\n";
            }
        }
        catch (const std::exception& e)
        {
            // std::cout << "---->poseExtractor::forwardPass catch" << "\n";
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<float> PoseExtractor::getHeatMapsCopy() const
    {
        // std::cout << "poseExtractor:: getHeatMapsCopy()\n";
        try
        {
            return spPoseExtractorNet->getHeatMapsCopy();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    std::vector<std::vector<std::array<float, 3>>> PoseExtractor::getCandidatesCopy() const
    {
        // std::cout << "poseExtractor:: getCandidatesCopy()\n";
        try
        {
            return spPoseExtractorNet->getCandidatesCopy();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::vector<std::vector<std::array<float,3>>>{};
        }
    }

    Array<float> PoseExtractor::getPoseKeypoints() const
    {
        // std::cout << "poseExtractor:: getPoseKeypoints()\n";
        try
        {
            return spPoseExtractorNet->getPoseKeypoints();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    Array<float> PoseExtractor::getPoseScores() const
    {
        // std::cout << "poseExtractor:: getPoseScores()\n";
        try
        {
            return spPoseExtractorNet->getPoseScores();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<float>{};
        }
    }

    float PoseExtractor::getScaleNetToOutput() const
    {
        // std::cout << "poseExtractor:: getScaleNetToOutput()\n";
        try
        {
            return spPoseExtractorNet->getScaleNetToOutput();
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return 0.;
        }
    }

    void PoseExtractor::keepTopPeople(Array<float>& poseKeypoints, const Array<float>& poseScores) const
    {
        // std::cout << "poseExtractor:: keepTopPeople(...)\n";
        try
        {
            // Keep only top N people
            if (spKeepTopNPeople)
                poseKeypoints = spKeepTopNPeople->keepTopPeople(poseKeypoints, poseScores);
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    Array<long long> PoseExtractor::extractIds(const Array<float>& poseKeypoints, const cv::Mat& cvMatInput,
                                               const unsigned long long imageViewIndex)
    {
        // std::cout << "poseExtractor:: extractIds(...)\n";
        try
        {
            // Run person ID extractor
            return (spPersonIdExtractor
                ? spPersonIdExtractor->extractIds(poseKeypoints, cvMatInput, imageViewIndex)
                : Array<long long>{poseKeypoints.getSize(0), -1});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    Array<long long> PoseExtractor::extractIdsLockThread(const Array<float>& poseKeypoints,
                                                         const cv::Mat& cvMatInput,
                                                         const unsigned long long imageViewIndex,
                                                         const long long frameId)
    {
        // std::cout << "poseExtractor:: extractIdsLockThread(...)\n";
        try
        {
            // Run person ID extractor
            return (spPersonIdExtractor
                ? spPersonIdExtractor->extractIdsLockThread(poseKeypoints, cvMatInput, imageViewIndex, frameId)
                : Array<long long>{poseKeypoints.getSize(0), -1});
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Array<long long>{};
        }
    }

    void PoseExtractor::track(Array<float>& poseKeypoints, Array<long long>& poseIds,
                              const cv::Mat& cvMatInput,
                              const unsigned long long imageViewIndex)
    {
        // std::cout << "poseExtractor:: track(...)\n";
        try
        {
            if (!spPersonTrackers->empty())
            {
                // Resize if required
                while (spPersonTrackers->size() <= imageViewIndex)
                    spPersonTrackers->emplace_back(std::make_shared<PersonTracker>(
                        (*spPersonTrackers)[0]->getMergeResults()));
                // Sanity check
                if (!poseKeypoints.empty() && poseIds.empty() && mNumberPeopleMax != 1)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Reset poseIds if keypoints is empty
                if (poseKeypoints.empty())
                    poseIds.reset();
                // Run person tracker
                if (spPersonTrackers->at(imageViewIndex))
                    (*spPersonTrackers)[imageViewIndex]->track(poseKeypoints, poseIds, cvMatInput);
                // Run person tracker
                (*spPersonTrackers)[imageViewIndex]->track(poseKeypoints, poseIds, cvMatInput);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractor::trackLockThread(Array<float>& poseKeypoints, Array<long long>& poseIds,
                                        const cv::Mat& cvMatInput,
                                        const unsigned long long imageViewIndex, const long long frameId)
    {
        // std::cout << "poseExtractor:: trackLockThread(...)\n";
        try
        {
            if (!spPersonTrackers->empty())
            {
                // Resize if required
                while (spPersonTrackers->size() <= imageViewIndex)
                    spPersonTrackers->emplace_back(std::make_shared<PersonTracker>(
                        (*spPersonTrackers)[0]->getMergeResults()));
                // Sanity check
                if (!poseKeypoints.empty() && poseIds.empty() && mNumberPeopleMax != 1)
                    error(errorMessage, __LINE__, __FUNCTION__, __FILE__);
                // Reset poseIds if keypoints is empty
                if (poseKeypoints.empty())
                    poseIds.reset();
                // Run person tracker
                if (spPersonTrackers->at(imageViewIndex))
                    (*spPersonTrackers)[imageViewIndex]->trackLockThread(
                        poseKeypoints, poseIds, cvMatInput, frameId);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
