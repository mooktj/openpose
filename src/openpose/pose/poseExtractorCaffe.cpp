#include <limits> // std::numeric_limits
#include <openpose/gpu/cuda.hpp>
#ifdef USE_CUDA
    #include <openpose/gpu/cuda.hu>
#endif
#include <openpose/pose/poseParameters.hpp>
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/keypoint.hpp>
#include <openpose/utilities/openCv.hpp>
#include <openpose/utilities/standard.hpp>
#include <openpose/pose/poseExtractorCaffe.hpp>

namespace op
{
    const bool TOP_DOWN_REFINEMENT = false; // Note: +5% acc 1 scale, -2% max acc setting

    #ifdef USE_CAFFE
        std::vector<ArrayCpuGpu<float>*> arraySharedToPtr(
            const std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeNetOutputBlob)
        {
            try
            {
                // Prepare spCaffeNetOutputBlobss
                std::vector<ArrayCpuGpu<float>*> caffeNetOutputBlobs(caffeNetOutputBlob.size());
                for (auto i = 0u ; i < caffeNetOutputBlobs.size() ; i++)
                    caffeNetOutputBlobs[i] = caffeNetOutputBlob[i].get();
                return caffeNetOutputBlobs;
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
                return {};
            }
        }

        inline void reshapePoseExtractorCaffe(
            std::shared_ptr<ResizeAndMergeCaffe<float>>& resizeAndMergeCaffe,
            std::shared_ptr<NmsCaffe<float>>& nmsCaffe,
            std::shared_ptr<BodyPartConnectorCaffe<float>>& bodyPartConnectorCaffe,
            std::shared_ptr<MaximumCaffe<float>>& maximumCaffe,
            std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeNetOutputBlobsShared,
            std::shared_ptr<ArrayCpuGpu<float>>& heatMapsBlob, std::shared_ptr<ArrayCpuGpu<float>>& peaksBlob,
            std::shared_ptr<ArrayCpuGpu<float>>& maximumPeaksBlob, const float scaleInputToNetInput,
            const PoseModel poseModel, const int gpuId, const float upsamplingRatio)
        {
            try
            {
                const auto netDescreaseFactor = (
                    upsamplingRatio <= 0.f ? getPoseNetDecreaseFactor(poseModel) : upsamplingRatio);
                // HeatMaps extractor blob and layer
                // Caffe modifies bottom - Heatmap gets resized
                const auto caffeNetOutputBlobs = arraySharedToPtr(caffeNetOutputBlobsShared);
                resizeAndMergeCaffe->Reshape(
                    caffeNetOutputBlobs, {heatMapsBlob.get()},
                    netDescreaseFactor, 1.f/scaleInputToNetInput, true, gpuId);
                // Pose extractor blob and layer
                nmsCaffe->Reshape({heatMapsBlob.get()}, {peaksBlob.get()}, getPoseMaxPeaks(),
                                  getPoseNumberBodyParts(poseModel), gpuId);
                // Pose extractor blob and layer
                bodyPartConnectorCaffe->Reshape({heatMapsBlob.get(), peaksBlob.get()}, gpuId);
                if (TOP_DOWN_REFINEMENT)
                    maximumCaffe->Reshape({heatMapsBlob.get()}, {maximumPeaksBlob.get()});
                // Cuda check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }

        void addCaffeNetOnThread(
            std::vector<std::shared_ptr<Net>>& net,
            std::vector<std::shared_ptr<ArrayCpuGpu<float>>>& caffeNetOutputBlob,
            const PoseModel poseModel, const int gpuId, const std::string& modelFolder,
            const std::string& protoTxtPath, const std::string& caffeModelPath, const bool enableGoogleLogging)
        {
            // std::cout << "poseExtractorCaffe:: addCaffeNetOnThread(...)\n";
            try
            {
                // Add Caffe Net
                net.emplace_back(
                    std::make_shared<NetCaffe>(
                        modelFolder + (protoTxtPath.empty() ? getPoseProtoTxt(poseModel) : protoTxtPath),
                        modelFolder + (caffeModelPath.empty() ? getPoseTrainedModel(poseModel) : caffeModelPath),
                        gpuId, enableGoogleLogging));
                // net.emplace_back(
                //     std::make_shared<NetOpenCv>(
                //         modelFolder + (protoTxtPath.empty() ? getPoseProtoTxt(poseModel) : protoTxtPath),
                //         modelFolder + (caffeModelPath.empty() ? getPoseTrainedModel(poseModel) : caffeModelPath),
                //         gpuId));
                // UNUSED(enableGoogleLogging);
                // Initializing them on the thread
                // std::cout << "---->protoTxtPath: " << protoTxtPath << "\n";

                net.back()->initializationOnThread();
                caffeNetOutputBlob.emplace_back((net.back().get())->getOutputBlobArray());
                // Sanity check
                if (net.size() != caffeNetOutputBlob.size())
                    error("Weird error, this should not happen. Notify us.", __LINE__, __FUNCTION__, __FILE__);
                // Cuda check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
            }
            catch (const std::exception& e)
            {
                error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            }
        }
    #endif

    PoseExtractorCaffe::PoseExtractorCaffe(
        const PoseModel poseModel, const std::string& modelFolder, const int gpuId,
        const std::vector<HeatMapType>& heatMapTypes, const ScaleMode heatMapScaleMode, const bool addPartCandidates,
        const bool maximizePositives, const std::string& protoTxtPath, const std::string& caffeModelPath,
        const float upsamplingRatio, const bool enableNet, const bool enableGoogleLogging) :

        PoseExtractorNet{poseModel, heatMapTypes, heatMapScaleMode, addPartCandidates, maximizePositives},
        mPoseModel{poseModel},
        mGpuId{gpuId},
        mModelFolder{modelFolder},
        mProtoTxtPath{protoTxtPath},
        mCaffeModelPath{caffeModelPath},
        mUpsamplingRatio{upsamplingRatio},
        mEnableNet{enableNet},
        mEnableGoogleLogging{enableGoogleLogging}
        #ifdef USE_CAFFE
            ,
            spResizeAndMergeCaffe{std::make_shared<ResizeAndMergeCaffe<float>>()},
            spNmsCaffe{std::make_shared<NmsCaffe<float>>()},
            spBodyPartConnectorCaffe{std::make_shared<BodyPartConnectorCaffe<float>>()},
            spMaximumCaffe{(TOP_DOWN_REFINEMENT ? std::make_shared<MaximumCaffe<float>>() : nullptr)}
        #endif
    {
        // std::cout << "poseExtractorCaffe:: PoseExtractorCaffe(...) constructor\n";
        try
        {
            // std::cout << "---->mProtoTxtPath: " << mProtoTxtPath << "\n";
            #ifdef USE_CAFFE
                // std::cout << "---->poseExtractorCaffe:: USE_CAFFE\n";
                // Layers parameters
                spBodyPartConnectorCaffe->setPoseModel(mPoseModel);
                spBodyPartConnectorCaffe->setMaximizePositives(maximizePositives);
            #else
                UNUSED(poseModel);
                UNUSED(modelFolder);
                UNUSED(gpuId);
                UNUSED(heatMapTypes);
                UNUSED(heatMapScaleMode);
                UNUSED(addPartCandidates);
                UNUSED(maximizePositives);
                UNUSED(protoTxtPath);
                UNUSED(caffeModelPath);
                UNUSED(enableGoogleLogging);
                error("OpenPose must be compiled with the `USE_CAFFE` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    PoseExtractorCaffe::~PoseExtractorCaffe()
    {
    }

    void PoseExtractorCaffe::netInitializationOnThread()
    {
        // std::cout << "poseExtractorCaffe:: netInitializationOnThread()\n";
        try
        {
            #ifdef USE_CAFFE
                if (mEnableNet)
                {
                    // Logging
                    log("Starting initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                    // Initialize Caffe net
                    addCaffeNetOnThread(
                        spNets, spCaffeNetOutputBlobs, mPoseModel, mGpuId,
                        mModelFolder, mProtoTxtPath, mCaffeModelPath,
                        mEnableGoogleLogging);
                    #ifdef USE_CUDA
                        cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    #endif
                }
                // Initialize blobs
                spHeatMapsBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                spPeaksBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                if (TOP_DOWN_REFINEMENT)
                    spMaximumPeaksBlob = {std::make_shared<ArrayCpuGpu<float>>(1,1,1,1)};
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                #endif
                // Logging
                log("Finished initialization on thread.", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
                // std::cout << "---->USE_CAFFE enabled! and netinitializationOnThread successful\n";
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    void PoseExtractorCaffe::forwardPass(
        const std::vector<Array<float>>& inputNetData, const Point<int>& inputDataSize,
        const std::vector<double>& scaleInputToNetInputs, const Array<float>& poseNetOutput)
    {
        // std::cout << "\n";
        // std::cout << "\n";
        // std::cout << "*****************************************************************************************************\n";
        // std::cout << "*****************************************************************************************************\n";
        // std::cout << "poseExtractorCaffe:: forwardPass(inputNetData, inputDataSize, scaleInputToNetInputs, poseNetOutput)\n";
        try
        {
            #ifdef USE_CAFFE
                // const auto REPS = 1;
                // double timeNormalize1 = 0.;
                // double timeNormalize2 = 0.;
                // double timeNormalize3 = 0.;
                // double timeNormalize4 = 0.;
                // OP_CUDA_PROFILE_INIT(REPS);

                // std::cout << "              ~~~~~~~~~~~~~print parameters~~~~~~~~~~~~~\n";
                // std::cout << "---->poseExtractorCaffe:: spNets = " << spNets.size() << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spResizeAndMergeCaffe = " << spResizeAndMergeCaffe.get()->mBottomSizes << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spNmsCaffe = " << spNmsCaffe << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spBodyPartConnectorCaffe = " << spBodyPartConnectorCaffe << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spMaximumCaffe = " << spMaximumCaffe << "\n";
                // std::cout << "---->poseExtractorCaffe:: mNetInput4DSizes.size() = " << mNetInput4DSizes.size() << "\n";


                // // std::cout << "---->poseExtractorCaffe:: spCaffeNetOutputBlobs = " << spCaffeNetOutputBlobs.size() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob shape_string() = " << spHeatMapsBlob.get()->shape_string() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num_axes() = " << spHeatMapsBlob.get()->num_axes() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob count() = " << spHeatMapsBlob.get()->count() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num() = " << spHeatMapsBlob.get()->num() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob channels() = " << spHeatMapsBlob.get()->channels() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob height() = " << spHeatMapsBlob.get()->height() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob width() = " << spHeatMapsBlob.get()->width() << "\n";

                // // std::cout << "---->poseExtractorCaffe:: spPeaksBlob = " << spPeaksBlob << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spMaximumPeaksBlob = " << spMaximumCaffe << "\n";
                // std::cout << "              ~~~~~~~~~~~~~~`~`~`~`~`~`~`~`~~~~~~~~~~~~~\n";

                // std::cout << "inputDataSize: " << inputDataSize << "\n";
                // Sanity checks
                if (inputNetData.empty())
                    error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                for (const auto& inputNetDataI : inputNetData)
                {
                    if (inputNetDataI.empty())
                        error("Empty inputNetData.", __LINE__, __FUNCTION__, __FILE__);
                }

                if (inputNetData.size() != scaleInputToNetInputs.size())
                    error("Size(inputNetData) must be same than size(scaleInputToNetInputs).",
                          __LINE__, __FUNCTION__, __FILE__);
                if (poseNetOutput.empty() != mEnableNet)
                {
                    const std::string errorMsg = ". Either use OpenPose default network (`--body 1`) or fill the"
                        " `poseNetOutput` argument (only 1 of those 2, not both).";
                    if (poseNetOutput.empty())
                        error("The argument poseNetOutput cannot be empty if mEnableNet is true" + errorMsg,
                              __LINE__, __FUNCTION__, __FILE__);
                    else
                        error("The argument poseNetOutput is not empty and you have also explicitly chosen to run"
                              " the OpenPose network" + errorMsg, __LINE__, __FUNCTION__, __FILE__);
                }

                // Resize std::vectors if required
                const auto numberScales = inputNetData.size();
                mNetInput4DSizes.resize(numberScales);

                // std::cout << "inputNetData.size(): " << inputNetData.size() << "\n";
                // std::cout << "numberScales: " << numberScales << "\n";
                // std::cout << "mNetInput4DSizes size(): " << mNetInput4DSizes.size() << "\n";


                // Process each image - Caffe deep network
                if (mEnableNet)
                {
                    while (spNets.size() < numberScales)
                        addCaffeNetOnThread(
                            spNets, spCaffeNetOutputBlobs, mPoseModel, mGpuId,
                            mModelFolder, mProtoTxtPath, mCaffeModelPath, false);

                    for (auto i = 0u ; i < inputNetData.size(); i++)
                    {
                        // std::cout << "------>forward:: spNets.at(i) where i = " << i << "\n";

                        // std::cout << "inputNetData[i].size(): " << sizeof(inputNetData[i])/sizeof(inputNetData[i][0]) << "\n";
                        spNets.at(i)->forwardPass(inputNetData[i]);

                        // std::cout << "--> spNets at i input #: " << spNets.at(i)->num_inputs() << "\n";
                        // std::cout << "--> spNets at i output #: " << spNets.at(i)->num_outputs() << "\n";
                    }
                    
                    // std::cout << "              ~~~~~~~~~~~~~print parameters~~~~~~~~~~~~~\n";
                    // std::cout << "---->poseExtractorCaffe:: spNets = " << spNets.size() << "\n";
                    // // std::cout << "---->poseExtractorCaffe:: spResizeAndMergeCaffe = " << spResizeAndMergeCaffe.get()->mBottomSizes << "\n";
                    // // std::cout << "---->poseExtractorCaffe:: spNmsCaffe = " << spNmsCaffe << "\n";
                    // // std::cout << "---->poseExtractorCaffe:: spBodyPartConnectorCaffe = " << spBodyPartConnectorCaffe << "\n";
                    // // std::cout << "---->poseExtractorCaffe:: spMaximumCaffe = " << spMaximumCaffe << "\n";
                    // std::cout << "---->poseExtractorCaffe:: mNetInput4DSizes.size() = " << mNetInput4DSizes.size() << "\n";


                    // // std::cout << "---->poseExtractorCaffe:: spCaffeNetOutputBlobs = " << spCaffeNetOutputBlobs.size() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob shape_string() = " << spHeatMapsBlob.get()->shape_string() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num_axes() = " << spHeatMapsBlob.get()->num_axes() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob count() = " << spHeatMapsBlob.get()->count() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num() = " << spHeatMapsBlob.get()->num() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob channels() = " << spHeatMapsBlob.get()->channels() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob height() = " << spHeatMapsBlob.get()->height() << "\n";
                    // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob width() = " << spHeatMapsBlob.get()->width() << "\n";

                    // // std::cout << "---->poseExtractorCaffe:: spPeaksBlob = " << spPeaksBlob << "\n";
                    // // std::cout << "---->poseExtractorCaffe:: spMaximumPeaksBlob = " << spMaximumCaffe << "\n";
                    // std::cout << "              ~~~~~~~~~~~~~~`~`~`~`~`~`~`~`~~~~~~~~~~~~~\n";

                }
                // If custom network output
                else
                {
                    // std::cout << "---->poseExtractorCaffe:: mEnableNet is false\n";
                    // Sanity check
                    if (inputNetData.size() != 1u)
                        error("Size(inputNetData) must match the provided heatmaps batch size ("
                              + std::to_string(inputNetData.size()) + " vs. " + std::to_string(1) + ").",
                              __LINE__, __FUNCTION__, __FILE__);
                    // Copy heatmap information
                    spCaffeNetOutputBlobs.clear();
                    const bool copyFromGpu = false;
                    spCaffeNetOutputBlobs.emplace_back(
                        std::make_shared<ArrayCpuGpu<float>>(poseNetOutput, copyFromGpu));
                }

                // std::cout << "---->poseExtractorCaffe: reshape blobs if required\n";
                // Reshape blobs if required
                for (auto i = 0u ; i < inputNetData.size(); i++)
                {
                    // std::cout << "------>poseExtractorCaffe:: reshape blobs if required where i = " << i << "\n";
                    // Reshape blobs if required - For dynamic sizes (e.g., images of different aspect ratio)
                    const auto changedVectors = !vectorsAreEqual(
                        mNetInput4DSizes.at(i), inputNetData[i].getSize());

                    // std::cout << "mNetInput4DSizes size(): " << mNetInput4DSizes.size() << "\n";
                    
                    // std::cout << "inputNetData[i].getSize(): " << typeid(inputNetData[i].getSize()).name() << "\n";
                    // std::cout << "-------->poseExtractorCaffe:: mNetInput4DSizes.at(i).size() = " << mNetInput4DSizes.at(i).size() << "\n";
                    // std::cout << "-------->poseExtractorCaffe:: inputNetData[i].getSize().size() = " << inputNetData[i].getSize().size() << "\n";

                    if (changedVectors)
                    {
                        // std::cout << "-------->poseExtractorCaffe:: changedVectors is true\n";

                        mNetInput4DSizes.at(i) = inputNetData[i].getSize();
                        reshapePoseExtractorCaffe(
                            spResizeAndMergeCaffe, spNmsCaffe, spBodyPartConnectorCaffe,
                            spMaximumCaffe, spCaffeNetOutputBlobs, spHeatMapsBlob,
                            spPeaksBlob, spMaximumPeaksBlob, 1.f, mPoseModel,
                            mGpuId, mUpsamplingRatio);
                            // In order to resize to input size to have same results as Matlab
                            // scaleInputToNetInputs[i] vs. 1.f
                        // std::cout << "mNetInput4DSizes.at(i) size: " << mNetInput4DSizes.at(i).size() << "\n";
                        // std::cout << "mNetInput4DSizes.at(i): " << mNetInput4DSizes.at(i).at(0) << "," << mNetInput4DSizes.at(i).at(1) << "," << mNetInput4DSizes.at(i).at(2) << "," << mNetInput4DSizes.at(i).at(3) << "\n";
                    }
                    // Get scale net to output (i.e., image input)
                    const auto ratio = (
                        mUpsamplingRatio <= 0.f
                            ? 1 : mUpsamplingRatio / getPoseNetDecreaseFactor(mPoseModel));
                    
                    // std::cout << "-------->poseExtractorCaffe:: mUpsamplingRatio = " << mUpsamplingRatio << "\n";
                    // std::cout << "-------->poseExtractorCaffe:: ratio = " << ratio << "\n";
                    // std::cout << "-------->poseExtractorCaffe:: mNetOutputSize.x = " << mNetOutputSize.x << ", mNetOutputSize.y = " << mNetOutputSize.y << "\n";
                    
                    if (changedVectors || TOP_DOWN_REFINEMENT)
                    {
                        // std::cout << "-------->poseExtractorCaffe:: changedVectors || TOP_DOWN_REFINEMENT\n";
                        
                        mNetOutputSize = Point<int>{
                            positiveIntRound(ratio*mNetInput4DSizes[0][3]),
                            positiveIntRound(ratio*mNetInput4DSizes[0][2])};
                        // std::cout << "mNetOutputSize: (" << mNetOutputSize.x << "," << mNetOutputSize.y << ")\n";
                        // std::cout << "mNetOutputSize data: " << mNetOutputSize << "\n";
                    }

                    // std::cout << "------>poseExtractorCaffe:: mNetOutputSize.x = " << mNetOutputSize.x << ", mNetOutputSize.y = " << mNetOutputSize.y << "\n";
                }

                // std::cout << "              ~~~~~~~~~~~~~print parameters~~~~~~~~~~~~~\n";
                // std::cout << "---->poseExtractorCaffe:: spNets = " << spNets.size() << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spResizeAndMergeCaffe = " << spResizeAndMergeCaffe.get()->mBottomSizes << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spNmsCaffe = " << spNmsCaffe << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spBodyPartConnectorCaffe = " << spBodyPartConnectorCaffe << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spMaximumCaffe = " << spMaximumCaffe << "\n";
                // std::cout << "---->poseExtractorCaffe:: mNetInput4DSizes.size() = " << mNetInput4DSizes.size() << "\n";


                // // std::cout << "---->poseExtractorCaffe:: spCaffeNetOutputBlobs = " << spCaffeNetOutputBlobs.size() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob shape_string() = " << spHeatMapsBlob.get()->shape_string() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num_axes() = " << spHeatMapsBlob.get()->num_axes() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob count() = " << spHeatMapsBlob.get()->count() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num() = " << spHeatMapsBlob.get()->num() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob channels() = " << spHeatMapsBlob.get()->channels() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob height() = " << spHeatMapsBlob.get()->height() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob width() = " << spHeatMapsBlob.get()->width() << "\n";

                // // std::cout << "---->poseExtractorCaffe:: spPeaksBlob = " << spPeaksBlob << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spMaximumPeaksBlob = " << spMaximumCaffe << "\n";
                // std::cout << "              ~~~~~~~~~~~~~~`~`~`~`~`~`~`~`~~~~~~~~~~~~~\n";

                // OP_CUDA_PROFILE_END(timeNormalize1, 1e3, REPS);
                // OP_CUDA_PROFILE_INIT(REPS);
                // 2. Resize heat maps + merge different scales
                // ~5ms (GPU) / ~20ms (CPU)
                const auto caffeNetOutputBlobs = arraySharedToPtr(spCaffeNetOutputBlobs);
                const std::vector<float> floatScaleRatios(scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());

                // std::cout << "caffeNetOutputBlobs num_axes: " << caffeNetOutputBlobs.at(0)->num_axes() << "\n";
                // std::cout << "caffeNetOutputBlobs channels: " << caffeNetOutputBlobs.at(0)->channels() << "\n";
                // std::cout << "caffeNetOutputBlobs count: " << caffeNetOutputBlobs.at(0)->count() << "\n";
                // std::cout << "caffeNetOutputBlobs height: " << caffeNetOutputBlobs.at(0)->height() << "\n";
                // std::cout << "caffeNetOutputBlobs width: " << caffeNetOutputBlobs.at(0)->width() << "\n";

                spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                spResizeAndMergeCaffe->Forward(caffeNetOutputBlobs, {spHeatMapsBlob.get()});
                // Get scale net to output (i.e., image input)
                // Note: In order to resize to input size, (un)comment the following lines
                const auto scaleProducerToNetInput = resizeGetScaleFactor(inputDataSize, mNetOutputSize);
                const Point<int> netSize{
                    positiveIntRound(scaleProducerToNetInput*inputDataSize.x),
                    positiveIntRound(scaleProducerToNetInput*inputDataSize.y)};
                mScaleNetToOutput = {(float)resizeGetScaleFactor(netSize, inputDataSize)};
                // mScaleNetToOutput = 1.f;
                // 3. Get peaks by Non-Maximum Suppression
                // std::cout << "---->poseExtractorCaffe: get peaks by non-maximum Suppression\n";
                // ~2ms (GPU) / ~7ms (CPU)
                // OP_CUDA_PROFILE_END(timeNormalize2, 1e3, REPS);
                const auto nmsThreshold = (float)get(PoseProperty::NMSThreshold);
                const auto nmsOffset = float(0.5/double(mScaleNetToOutput));
                // OP_CUDA_PROFILE_INIT(REPS);
                spNmsCaffe->setThreshold(nmsThreshold);
                spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
                spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()}); 
                // std::cout << "????spNmsCaffe->Forward(..)\n";
                // 4. Connecting body parts
                // std::cout << "---->poseExtractorCaffe: connecting body parts\n";
                // OP_CUDA_PROFILE_END(timeNormalize3, 1e3, REPS);
                // OP_CUDA_PROFILE_INIT(REPS);
                spBodyPartConnectorCaffe->setScaleNetToOutput(mScaleNetToOutput);
                spBodyPartConnectorCaffe->setInterMinAboveThreshold(
                    (float)get(PoseProperty::ConnectInterMinAboveThreshold));
                spBodyPartConnectorCaffe->setInterThreshold((float)get(PoseProperty::ConnectInterThreshold));
                spBodyPartConnectorCaffe->setMinSubsetCnt((int)get(PoseProperty::ConnectMinSubsetCnt));
                spBodyPartConnectorCaffe->setMinSubsetScore((float)get(PoseProperty::ConnectMinSubsetScore));
                // Note: BODY_25D will crash (only implemented for CPU version)
                spBodyPartConnectorCaffe->Forward(
                    {spHeatMapsBlob.get(), spPeaksBlob.get()}, mPoseKeypoints, mPoseScores);
                // OP_CUDA_PROFILE_END(timeNormalize4, 1e3, REPS);
                // log("1 = " + std::to_string(timeNormalize1) + " msecs.");
                // log("2 = " + std::to_string(timeNormalize2) + " msecs.");
                // log("3 = " + std::to_string(timeNormalize3) + " msecs.");
                // log("4 = " + std::to_string(timeNormalize4) + " msecs.");
                // Re-run on each person
                // std::cout << "---->poseExtractorCaffe: re-run on each person\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob = " << spHeatMapsBlob.get()->channels() << "\n";
                
                // std::cout << "---------------------->poseExtractorCaffe forwardPass done 1\n";

                if (TOP_DOWN_REFINEMENT)
                {
                    // std::cout << "------>poseExtractorCaffe: TOP_DOWN_REFINEMENT\n";
                    // Get each person rectangle
                    for (auto person = 0 ; person < mPoseKeypoints.getSize(0) ; person++)
                    {
                        // Get person rectangle resized to input size
                        // std::cout << "-------->poseExtractorCaffe: get person rectangle resized to input size\n";
                        const auto rectangleF = getKeypointsRectangle(mPoseKeypoints, person, nmsThreshold)
                                              / mScaleNetToOutput;
                        // Make rectangle bigger to make sure the whole body is inside
                        cv::Rect cvRectangle{
                            positiveIntRound(rectangleF.x - 0.2*rectangleF.width),
                            positiveIntRound(rectangleF.y - 0.2*rectangleF.height),
                            positiveIntRound(rectangleF.width*1.4),
                            positiveIntRound(rectangleF.height*1.4)
                        };
                        keepRoiInside(cvRectangle, inputNetData[0].getSize(3), inputNetData[0].getSize(2));
                        // Input size
                        // // Note: In order to preserve speed but maximize accuracy
                        // // If e.g. rectange = 10x1 and inputSize = 656x368 --> targetSize = 656x368
                        // // Note: If e.g. rectange = 1x10 and inputSize = 656x368 --> targetSize = 368x656
                        // const auto width = ( ? cvRectangle.width : cvRectangle.height);
                        // const auto height = (width == cvRectangle.width ? cvRectangle.height : cvRectangle.width);
                        // const Point<int> inputSize{width, height};
                        // Note: If inputNetData.size = -1x368 --> TargetSize = 368x-1
                        const Point<int> inputSizeInit{cvRectangle.width, cvRectangle.height};
                        // Target size
                        Point<int> targetSize;
                        // Optimal case (using training size)
                        if (inputNetData[0].getSize(2) >= 368 || inputNetData[0].getVolume(2,3) >= 135424) // 368^2
                            targetSize = Point<int>{368, 368};
                        // Low resolution cases: Keep same area than biggest scale
                        else
                        {
                            const auto minSide = fastMin(
                                368, fastMin(inputNetData[0].getSize(2), inputNetData[0].getSize(3)));
                            const auto maxSide = fastMin(
                                368, fastMax(inputNetData[0].getSize(2), inputNetData[0].getSize(3)));
                            // Person bounding box is vertical
                            if (cvRectangle.width < cvRectangle.height)
                                targetSize = Point<int>{minSide, maxSide};
                            // Person bounding box is horizontal
                            else
                                targetSize = Point<int>{maxSide, minSide};
                        }
                        // Fill resizedImage
                        /*const*/ auto scaleNetToRoi = resizeGetScaleFactor(inputSizeInit, targetSize);
                        // Update rectangle to avoid black padding and instead take full advantage of the network area
                        const auto padding = Point<int>{
                            (int)std::round((targetSize.x-1) / scaleNetToRoi + 1 - inputSizeInit.x),
                            (int)std::round((targetSize.y-1) / scaleNetToRoi + 1 - inputSizeInit.y)
                        };
                        // Width requires padding
                        if (padding.x > 2 || padding.y > 2) // 2 pixels as threshold
                        {
                            if (padding.x > 2) // 2 pixels as threshold
                            {
                                cvRectangle.x -= padding.x/2;
                                cvRectangle.width += padding.x;
                            }
                            else if (padding.y > 2) // 2 pixels as threshold
                            {
                                cvRectangle.y -= padding.y/2;
                                cvRectangle.height += padding.y;
                            }
                            keepRoiInside(cvRectangle, inputNetData[0].getSize(3), inputNetData[0].getSize(2));
                            scaleNetToRoi = resizeGetScaleFactor(
                                Point<int>{cvRectangle.width, cvRectangle.height}, targetSize);
                        }
                        // std::cout << "------>poseExtractorCaffe:: scaleNetToRoi = " << scaleNetToRoi << "\n";
                        // No if scaleNetToRoi < 1 (image would be shrinked, so we assume best result already obtained)
                        if (scaleNetToRoi > 1)
                        {
                            // std::cout << "---!!!!!!!!!!!!!!!!!!!!!!!!!!!!!scaleNetToRoi---!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
                            const auto areaInput = inputNetData[0].getVolume(2,3);
                            const auto areaRoi = targetSize.area();
                            Array<float> inputNetDataRoi{{1, 3, targetSize.y, targetSize.x}};
                            for (auto c = 0u ; c < 3u ; c++)
                            {
                                // Input image
                                const cv::Mat wholeInputCvMat(
                                    inputNetData[0].getSize(2), inputNetData[0].getSize(3), CV_32FC1,
                                    inputNetData[0].getPseudoConstPtr() + c * areaInput);
                                // Input image cropped
                                const cv::Mat inputCvMat(wholeInputCvMat, cvRectangle);
                                // Resize image for inputNetDataRoi
                                cv::Mat resizedImageCvMat(
                                    inputNetDataRoi.getSize(2), inputNetDataRoi.getSize(3), CV_32FC1,
                                    inputNetDataRoi.getPtr() + c * areaRoi);
                                resizeFixedAspectRatio(resizedImageCvMat, inputCvMat, scaleNetToRoi, targetSize);
                            }

                            // std::cout << "---->poseExtractorCaffe: re-process image ??why-------------------------------------------\n";
                            // Re-Process image
                            // 1. Caffe deep network
                            spNets.at(0)->forwardPass(inputNetDataRoi);
                            std::vector<std::shared_ptr<ArrayCpuGpu<float>>> caffeNetOutputBlob{
                                spCaffeNetOutputBlobs[0]};
                            // Reshape blobs
                            if (!vectorsAreEqual(mNetInput4DSizes.at(0), inputNetDataRoi.getSize()))
                            {
                                mNetInput4DSizes.at(0) = inputNetDataRoi.getSize();
                                reshapePoseExtractorCaffe(
                                    spResizeAndMergeCaffe, spNmsCaffe,
                                    spBodyPartConnectorCaffe, spMaximumCaffe,
                                    // spCaffeNetOutputBlobs,
                                    caffeNetOutputBlob, spHeatMapsBlob, spPeaksBlob,
                                    spMaximumPeaksBlob, 1.f, mPoseModel, mGpuId,
                                    mUpsamplingRatio);
                            }
                            // 2. Resize heat maps + merge different scales
                            const auto caffeNetOutputBlobs = arraySharedToPtr(caffeNetOutputBlob);
                            // const std::vector<float> floatScaleRatios(
                            //     scaleInputToNetInputs.begin(), scaleInputToNetInputs.end());
                            const std::vector<float> floatScaleRatios{(float)scaleInputToNetInputs[0]};
                            spResizeAndMergeCaffe->setScaleRatios(floatScaleRatios);
                            spResizeAndMergeCaffe->Forward(
                                caffeNetOutputBlobs, {spHeatMapsBlob.get()});
                            // Get scale net to output (i.e., image input)
                            const auto scaleRoiToOutput = float(mScaleNetToOutput / scaleNetToRoi);
                            // 3. Get peaks by Non-Maximum Suppression
                            const auto nmsThresholdRefined = 0.02f;
                            spNmsCaffe->setThreshold(nmsThresholdRefined);
                            const auto nmsOffset = float(0.5/double(scaleRoiToOutput));
                            spNmsCaffe->setOffset(Point<float>{nmsOffset, nmsOffset});
                            spNmsCaffe->Forward({spHeatMapsBlob.get()}, {spPeaksBlob.get()});
                            // Define poseKeypoints
                            Array<float> poseKeypoints;
                            Array<float> poseScores;
                            // 4. Connecting body parts
                            // Get scale net to output (i.e., image input)
                            spBodyPartConnectorCaffe->setScaleNetToOutput(scaleRoiToOutput);
                            spBodyPartConnectorCaffe->setInterThreshold(0.01f);
                            spBodyPartConnectorCaffe->Forward(
                                {spHeatMapsBlob.get(), spPeaksBlob.get()}, poseKeypoints, poseScores);
                            // If detected people in new subnet
                            if (!poseKeypoints.empty())
                            {
                                // // Scale back keypoints
                                const auto xOffset = float(cvRectangle.x*mScaleNetToOutput);
                                const auto yOffset = float(cvRectangle.y*mScaleNetToOutput);
                                scaleKeypoints2d(poseKeypoints, 1.f, 1.f, xOffset, yOffset);
                                // Re-assign person back
                                // // Option a) Just use biggest person (simplest but fails with crowded people)
                                // const auto personRefined = getBiggestPerson(poseKeypoints, nmsThreshold);
                                // Option b) Get minimum keypoint distance
                                // Get min distance
                                int personRefined = -1;
                                float personAverageDistance = std::numeric_limits<float>::max();
                                for (auto person2 = 0 ; person2 < poseKeypoints.getSize(0) ; person2++)
                                {
                                    // Get average distance
                                    const auto currentAverageDistance = getDistanceAverage(
                                        mPoseKeypoints, person, poseKeypoints, person2, nmsThreshold);
                                    // Update person
                                    if (personAverageDistance > currentAverageDistance
                                        && getNonZeroKeypoints(poseKeypoints, person2, nmsThreshold)
                                            >= 0.75*getNonZeroKeypoints(mPoseKeypoints, person, nmsThreshold))
                                    {
                                        personRefined = person2;
                                        personAverageDistance = currentAverageDistance;
                                    }
                                }
                                // Get max ROI
                                int personRefinedRoi = -1;
                                float personRoi = -1.f;
                                for (auto person2 = 0 ; person2 < poseKeypoints.getSize(0) ; person2++)
                                {
                                    // Get ROI
                                    const auto currentRoi = getKeypointsRoi(
                                        mPoseKeypoints, person, poseKeypoints, person2, nmsThreshold);
                                    // Update person
                                    if (personRoi < currentRoi
                                        && getNonZeroKeypoints(poseKeypoints, person2, nmsThreshold)
                                            >= 0.75*getNonZeroKeypoints(mPoseKeypoints, person, nmsThreshold))
                                    {
                                        personRefinedRoi = person2;
                                        personRoi = currentRoi;
                                    }
                                }
                                // If good refined candidate found
                                // I.e., if both max ROI and min dist match on same person id
                                if (personRefined == personRefinedRoi && personRefined > -1)
                                {
                                    // Update only if avg dist is small enough
                                    const auto personRectangle = getKeypointsRectangle(
                                        mPoseKeypoints, person, nmsThreshold);
                                    const auto personRatio = 0.1f * (float)std::sqrt(
                                        personRectangle.x*personRectangle.x + personRectangle.y*personRectangle.y);
                                    // if (mPoseScores[person] < poseScores[personRefined]) // This harms accuracy
                                    if (personAverageDistance < personRatio)
                                    {
                                        const auto personArea = mPoseKeypoints.getVolume(1,2);
                                        const auto personIndex = person * personArea;
                                        const auto personRefinedIndex = personRefined * personArea;
                                        // mPoseKeypoints: Update keypoints
                                        // Option a) Using refined ones
                                        std::copy(
                                            poseKeypoints.getPtr() + personRefinedIndex,
                                            poseKeypoints.getPtr() + personRefinedIndex + personArea,
                                            mPoseKeypoints.getPtr() + personIndex);
                                        mPoseScores[person] = poseScores[personRefined];
                                        // // Option b) Using ones with highest score (-6% acc single scale)
                                        // // Fill gaps
                                        // for (auto part = 0 ; part < mPoseKeypoints.getSize(1) ; part++)
                                        // {
                                        //     // For currently empty keypoints
                                        //     const auto partIndex = personIndex+3*part;
                                        //     const auto partRefinedIndex = personRefinedIndex+3*part;
                                        //     const auto scoreDifference = poseKeypoints[partRefinedIndex+2]
                                        //                                - mPoseKeypoints[partIndex+2];
                                        //     if (scoreDifference > 0)
                                        //     {
                                        //         const auto x = poseKeypoints[partRefinedIndex];
                                        //         const auto y = poseKeypoints[partRefinedIndex + 1];
                                        //         mPoseKeypoints[partIndex] = x;
                                        //         mPoseKeypoints[partIndex+1] = y;
                                        //         mPoseKeypoints[partIndex+2] += scoreDifference;
                                        //         mPoseScores[person] += scoreDifference;
                                        //     }
                                        // }

                                        // No acc improvement (-0.05% acc single scale)
                                        // // Finding all missing peaks (CPM-style)
                                        // // Only if no other person in there (otherwise 2% accuracy drop)
                                        // if (getNonZeroKeypoints(mPoseKeypoints, person, nmsThresholdRefined) > 0)
                                        // {
                                        //     // Get whether 0% ROI with other people
                                        //     // Get max ROI
                                        //     bool overlappingPerson = false;
                                        //     for (auto person2 = 0 ; person2 < mPoseKeypoints.getSize(0) ; person2++)
                                        //     {
                                        //         if (person != person2)
                                        //         {
                                        //             // Get ROI
                                        //             const auto currentRoi = getKeypointsRoi(
                                        //                 mPoseKeypoints, person, person2, nmsThreshold);
                                        //             // Update person
                                        //             if (currentRoi > 0.f)
                                        //             {
                                        //                 overlappingPerson = true;
                                        //                 break;
                                        //             }
                                        //         }
                                        //     }
                                        //     if (!overlappingPerson)
                                        //     {
                                        //         // Get keypoint with maximum probability per channel
                                        //         spMaximumCaffe->Forward(
                                        //             {spHeatMapsBlob.get()}, {spMaximumPeaksBlob.get()});
                                        //         // Fill gaps
                                        //         const auto* posePeaksPtr = spMaximumPeaksBlob->mutable_cpu_data();
                                        //         for (auto part = 0 ; part < mPoseKeypoints.getSize(1) ; part++)
                                        //         {
                                        //             // For currently empty keypoints
                                        //             if (mPoseKeypoints[personIndex+3*part+2] < nmsThresholdRefined)
                                        //             {
                                        //                 const auto xyIndex = 3*part;
                                        //                 const auto x = posePeaksPtr[xyIndex]*scaleRoiToOutput + xOffset;
                                        //                 const auto y = posePeaksPtr[xyIndex + 1]*scaleRoiToOutput + yOffset;
                                        //                 const auto rectangle = getKeypointsRectangle(
                                        //                     mPoseKeypoints, person, nmsThresholdRefined);
                                        //                 if (x >= rectangle.x && x < rectangle.x + rectangle.width
                                        //                     && y >= rectangle.y && y < rectangle.y + rectangle.height)
                                        //                 {
                                        //                     const auto score = posePeaksPtr[xyIndex + 2];
                                        //                     const auto baseIndex = personIndex + 3*part;
                                        //                     mPoseKeypoints[baseIndex] = x;
                                        //                     mPoseKeypoints[baseIndex+1] = y;
                                        //                     mPoseKeypoints[baseIndex+2] = score;
                                        //                     mPoseScores[person] += score;
                                        //                 }
                                        //             }
                                        //         }
                                        //     }
                                        // }
                                    }
                                }
                            }
                        }
                    }
                }
                
                // std::cout << "              ~~~~~~~~~~~~~print parameters~~~~~~~~~~~~~\n";
                // std::cout << "---->poseExtractorCaffe:: spNets = " << spNets.size() << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spResizeAndMergeCaffe = " << spResizeAndMergeCaffe.get()->mBottomSizes << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spNmsCaffe = " << spNmsCaffe << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spBodyPartConnectorCaffe = " << spBodyPartConnectorCaffe << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spMaximumCaffe = " << spMaximumCaffe << "\n";
                // std::cout << "---->poseExtractorCaffe:: mNetInput4DSizes.size() = " << mNetInput4DSizes.size() << "\n";


                // // std::cout << "---->poseExtractorCaffe:: spCaffeNetOutputBlobs = " << spCaffeNetOutputBlobs.size() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob shape_string() = " << spHeatMapsBlob.get()->shape_string() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num_axes() = " << spHeatMapsBlob.get()->num_axes() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob count() = " << spHeatMapsBlob.get()->count() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob num() = " << spHeatMapsBlob.get()->num() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob channels() = " << spHeatMapsBlob.get()->channels() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob height() = " << spHeatMapsBlob.get()->height() << "\n";
                // std::cout << "---->poseExtractorCaffe:: spHeatMapsBlob width() = " << spHeatMapsBlob.get()->width() << "\n";

                // // std::cout << "---->poseExtractorCaffe:: spPeaksBlob = " << spPeaksBlob << "\n";
                // // std::cout << "---->poseExtractorCaffe:: spMaximumPeaksBlob = " << spMaximumCaffe << "\n";
                // std::cout << "              ~~~~~~~~~~~~~~`~`~`~`~`~`~`~`~~~~~~~~~~~~~\n";

                // // std::cout << "---->poseExtractorCaffe:: mNetInput4DSizes.at(0).at(0) = " << (mNetInput4DSizes.at(0)).at(0) << "\n";
                // std::cout << "---------------------->poseExtractorCaffe forwardPass done 2\n";
                // std::cout << "*****************************************************************************************************\n";
                // std::cout << "*****************************************************************************************************\n";
                // std::cout << "\n";
                // std::cout << "\n";

                // 5. CUDA sanity check
                #ifdef USE_CUDA
                    cudaCheck(__LINE__, __FUNCTION__, __FILE__);
                    // std::cout << "---->poseExtractorCaffe -> forwardPass(..):: USE_CUDA enabled!\n";
                #endif
                    // std::cout << "---->poseExtractorCaffe -> forwardPass(..):: USE_CAFFE enabled! and forwardPass successful\n";
            #else
                UNUSED(inputNetData);
                UNUSED(inputDataSize);
                UNUSED(scaleInputToNetInputs);
            #endif
            // std::cout << "poseExtractorCaffe:: forwardPass() done......................\n";
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    const float* PoseExtractorCaffe::getCandidatesCpuConstPtr() const
    {
        // std::cout << "poseExtractorCaffe:: getCandidatesCpuConstPtr()\n";
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spPeaksBlob->cpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getCandidatesGpuConstPtr() const
    {
        // std::cout << "poseExtractorCaffe:: getCandidatesGpuConstPtr()\n";
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spPeaksBlob->gpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getHeatMapCpuConstPtr() const
    {
        // std::cout << "poseExtractorCaffe:: getHeatMapCpuConstPtr()\n";
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spHeatMapsBlob->cpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    const float* PoseExtractorCaffe::getHeatMapGpuConstPtr() const
    {
        // std::cout << "poseExtractorCaffe:: getHeatMapGpuConstPtr()\n";
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spHeatMapsBlob->gpu_data();
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }

    std::vector<int> PoseExtractorCaffe::getHeatMapSize() const
    {
        // std::cout << "poseExtractorCaffe:: getHeatMapSize()\n";
        try
        {
            #ifdef USE_CAFFE
                checkThread();
                return spHeatMapsBlob->shape();
            #else
                return {};
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    const float* PoseExtractorCaffe::getPoseGpuConstPtr() const
    {
        // std::cout << "getPoseGpuConstPtr()\n";
        try
        {
            #ifdef USE_CAFFE
                error("GPU pointer for people pose data not implemented yet.", __LINE__, __FUNCTION__, __FILE__);
                checkThread();
                return nullptr;
            #else
                return nullptr;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return nullptr;
        }
    }
}
