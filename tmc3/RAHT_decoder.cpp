/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2017-2018, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "RAHT.h"

using namespace pcc::RAHT;

namespace pcc {
//============================================================================
// remove any non-unique leaves from a level in the uraht tree

int
reduceUnique_decoder(
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut)
{
  // process a single level of the tree
  int64_t posPrev = -1;
  auto weightsInWrIt = weightsIn->begin();
  auto weightsInRdIt = weightsIn->cbegin();
  for (int i = 0; i < numNodes; i++) {
    const auto& node = *weightsInRdIt++;

    // copy across unique nodes
    if (node.pos != posPrev) {
      posPrev = node.pos;
      *weightsInWrIt++ = node;
      continue;
    }

    // duplicate node
    (weightsInWrIt - 1)->weight += node.weight;
    weightsOut->push_back(node);
  }

  // number of nodes in next level
  return std::distance(weightsIn->begin(), weightsInWrIt);
}

//============================================================================
// Split a level of values into sum and difference pairs.

int
reduceLevel_decoder(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,
  std::vector<UrahtNode>* weightsOut)
{
  // process a single level of the tree
  int64_t posPrev = -1;
  auto weightsInWrIt = weightsIn->begin();
  auto weightsInRdIt = weightsIn->cbegin();

  for (int i = 0; i < numNodes; i++) {
    auto& node = *weightsInRdIt++;
    bool newPair = (posPrev ^ node.pos) >> level != 0;
    posPrev = node.pos;
    if (newPair) {
      *weightsInWrIt++ = node;
    } 
	else {
      auto& left = *(weightsInWrIt - 1);
      left.weight += node.weight;
      left.qp[0] = (left.qp[0] + node.qp[0]) >> 1;
      left.qp[1] = (left.qp[1] + node.qp[1]) >> 1;
      weightsOut->push_back(node);
    }
  }

  // number of nodes in next level
  return std::distance(weightsIn->begin(), weightsInWrIt);
}

//============================================================================
// Merge sum and difference values to form a tree.

void
expandLevel_decoder(
  int level,
  int numNodes,
  std::vector<UrahtNode>* weightsIn,  // expand by numNodes before expand
  std::vector<UrahtNode>* weightsOut  // shrink after expand
)
{
  if (numNodes == 0)
    return;

  // process a single level of the tree
  auto weightsInWrIt = weightsIn->rbegin();
  auto weightsInRdIt = std::next(weightsIn->crbegin(), numNodes);
  auto weightsOutRdIt = weightsOut->crbegin();

  for (int i = 0; i < numNodes;) {
    bool isPair = (weightsOutRdIt->pos ^ weightsInRdIt->pos) >> level == 0;
    if (!isPair) {
      *weightsInWrIt++ = *weightsInRdIt++;
      
      continue;
    }

    // going to process a pair
    i++;

    // Out node is inserted before In node.
    const auto& nodeDelta = *weightsInWrIt++ = *weightsOutRdIt++;

    // move In node to correct position, subtracting delta
    *weightsInWrIt = *weightsInRdIt++;
    (weightsInWrIt++)->weight -= nodeDelta.weight;
  }
}

//============================================================================
// Generate the spatial prediction of a block.

template<bool haarFlag, int numAttrs, bool rahtExtension, typename It>
void
intraDcPred_decoder(
  const int parentNeighIdx[19],
  const int childNeighIdx[12][8],
  int occupancy,
  It first,
  It firstChild,
  FixedPoint predBuf[][8],
  const RahtPredictionParams &rahtPredParams, 
  int64_t& limitLow,
  int64_t& limitHigh)
{
  static const uint8_t predMasks[19] = {255, 240, 204, 170, 192, 160, 136,
                                        3,   5,   15,  17,  51,  85,  10,
                                        34,  12,  68,  48,  80};

  const auto& predWeightParent = rahtPredParams.predWeightParent;
  const auto& predWeightChild = rahtPredParams.predWeightChild;

  static const int kDivisors[64] = {
    32768, 16384, 10923, 8192, 6554, 5461, 4681, 4096, 3641, 3277, 2979,
    2731,  2521,  2341,  2185, 2048, 1928, 1820, 1725, 1638, 1560, 1489,
    1425,  1365,  1311,  1260, 1214, 1170, 1130, 1092, 1057, 1024, 993,
    964,   936,   910,   886,  862,  840,  819,  799,  780,  762,  745,
    728,   712,   697,   683,  669,  655,  643,  630,  618,  607,  596,
    585,   575,   565,   555,  546,  537,  529,  520,  512};

  int weightSum[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

  
  int64_t neighValue[3];
  int64_t childNeighValue[3];
  int64_t intraChildNeighValue[3];

  const auto parentOnlyCheckMaxIdx =
    rahtPredParams.raht_subnode_prediction_enabled_flag ? 7 : 19;
  for (int i = 0; i < parentOnlyCheckMaxIdx; i++) {
    if (parentNeighIdx[i] == -1)
      continue;

    auto neighValueIt = std::next(first, numAttrs * parentNeighIdx[i]);
    for (int k = 0; k < numAttrs; k++)
      neighValue[k] = *neighValueIt++;

    // skip neighbours that are outside of threshold limits
    if (i) {
      if (10 * neighValue[0] <= limitLow || 10 * neighValue[0] >= limitHigh)
        continue;
    } else {
      constexpr int ratioThreshold1 = 2;
      constexpr int ratioThreshold2 = 25;
      limitLow = ratioThreshold1 * neighValue[0];
      limitHigh = ratioThreshold2 * neighValue[0];
    }

    // apply weighted neighbour value to masked positions
    for (int k = 0; k < numAttrs; k++)
      if (rahtExtension)
        neighValue[k] *= predWeightParent[i];
      else
        neighValue[k] *= predWeightParent[i] << pcc::FixedPoint::kFracBits;

    int mask = predMasks[i] & occupancy;
    for (int j = 0; mask; j++, mask >>= 1) {
      if (mask & 1) {
        weightSum[j] += predWeightParent[i];
        for (int k = 0; k < numAttrs; k++) {
          predBuf[k][j].val += neighValue[k];
        }
      }
    }
  }
  if (rahtPredParams.raht_subnode_prediction_enabled_flag) {
    for (int i = 0; i < 12; i++) {
      if (parentNeighIdx[7 + i] == -1)
        continue;

      auto neighValueIt = std::next(first, numAttrs * parentNeighIdx[7 + i]);
      for (int k = 0; k < numAttrs; k++)
        neighValue[k] = *neighValueIt++;

      // skip neighbours that are outside of threshold limits
      if (10 * neighValue[0] <= limitLow || 10 * neighValue[0] >= limitHigh)
        continue;

      // apply weighted neighbour value to masked positions
      for (int k = 0; k < numAttrs; k++)
        if (rahtExtension)
          neighValue[k] *= predWeightParent[7 + i];
        else
          neighValue[k] *= predWeightParent[7 + i] << pcc::FixedPoint::kFracBits;

      int mask = predMasks[7 + i] & occupancy;
      for (int j = 0; mask; j++, mask >>= 1) {
        if (mask & 1) {
          if (childNeighIdx[i][j] != -1) {
            weightSum[j] += predWeightChild[i];
            auto childNeighValueIt =
              std::next(firstChild, numAttrs * childNeighIdx[i][j]);
            for (int k = 0; k < numAttrs; k++)
              if (rahtExtension)
                childNeighValue[k] = (*childNeighValueIt++)
                  * predWeightChild[i];
              else
                childNeighValue[k] = (*childNeighValueIt++)
                  * (predWeightChild[i] << pcc::FixedPoint::kFracBits);

            for (int k = 0; k < numAttrs; k++)
              predBuf[k][j].val += childNeighValue[k];

          } else {
            weightSum[j] += predWeightParent[7 + i];
            for (int k = 0; k < numAttrs; k++) {
              predBuf[k][j].val += neighValue[k];
            }
          }
        }
      }
    }
  }

  // normalise
  FixedPoint div;
  for (int i = 0; i < 8; i++, occupancy >>= 1) {
    if (occupancy & 1) {
      div.val = kDivisors[weightSum[i]];
      for (int k = 0; k < numAttrs; k++) {
        predBuf[k][i] *= div;
      }
      if (haarFlag) {
        for (int k = 0; k < numAttrs; k++) {
          predBuf[k][i].val = (predBuf[k][i].val >> predBuf[k][i].kFracBits)
            << predBuf[k][i].kFracBits;
        }
      }
    }
  }
}

//============================================================================
// Core transform process (for encoder/decoder)
template<bool haarFlag, int numAttrs, bool rahtExtension, class ModeCoder>
void
uraht_process_decoder(
  const RahtPredictionParams& rahtPredParams,
  const QpSet& qpset,
  const Qps* pointQpOffsets,
  int numPoints,
  int64_t* positions,
  int* attributes,
  int32_t* coeffBufIt,
  AttributeInterPredParams& attrInterPredParams,
  ModeCoder& coder)
{
  // coefficients are stored in three planar arrays.  coeffBufItK is a set
  // of iterators to each array.
  int32_t* coeffBufItK[3] = {
    coeffBufIt, coeffBufIt + numPoints, coeffBufIt + numPoints * 2};

  // early termination only one point
  if (numPoints == 1) {
    auto quantizers = qpset.quantizers(0, pointQpOffsets[0]);
    for (int k = 0; k < numAttrs; k++) {
      auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];
	  int64_t coeff = *coeffBufItK[k]++;
      attributes[k] =
        divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);
    }
    return;
  }

  std::vector<UrahtNode> weightsLf, weightsHf;
  std::vector<int64_t> attrsLf, attrsHf;

  std::vector<UrahtNode> weightsLf_ref, weightsHf_ref;
  std::vector<int64_t> attrsLf_ref, attrsHf_ref;

  bool enableLCPPred = rahtPredParams.raht_last_component_prediction_enabled_flag;

  bool enableACInterPred = attrInterPredParams.enableAttrInterPred;

  bool enableACRDOInterPred =
    attrInterPredParams.paramsForInterRAHT.raht_enable_inter_intra_layer_RDO
    && enableACInterPred;

  bool enableACRDONonPred = rahtPredParams.raht_enable_intraPred_nonPred_code_layer
    && rahtPredParams.raht_prediction_enabled_flag;

  int treeDepth = 0;
  int treeDepthLimit = 1 + attrInterPredParams.paramsForInterRAHT.raht_inter_prediction_depth_minus1;

  bool enableFilterEstimation = attrInterPredParams.paramsForInterRAHT.enableFilterEstimation;
  std::vector<int64_t> fixedFilterTaps = {128, 128, 128, 127, 125, 121, 115};
  int skipInitLayersForFiltering = attrInterPredParams.paramsForInterRAHT.skipInitLayersForFiltering;

  weightsLf.reserve(numPoints);
  attrsLf.reserve(numPoints * numAttrs);

  int regionQpShift = 4;
  const int maxAcCoeffQpOffsetLayers = qpset.rahtAcCoeffQps.size() - 1;

  // copy positions into internal form
  // no need to copy attribute at decoder side
  for (int i = 0; i < numPoints; i++) {
    UrahtNode node;
    node.pos = positions[i];
	node.weight = 1;
	node.qp = {
      int16_t(pointQpOffsets[i][0] << regionQpShift),
      int16_t(pointQpOffsets[i][1] << regionQpShift)};
    weightsLf.emplace_back(node);
  }

  weightsHf.reserve(numPoints);
  attrsHf.reserve(numPoints * numAttrs);

  if (enableACInterPred) {
    weightsLf_ref.reserve(attrInterPredParams.paramsForInterRAHT.voxelCount);
    attrsLf_ref.reserve(attrInterPredParams.paramsForInterRAHT.voxelCount * numAttrs);

    for (int i = 0; i < attrInterPredParams.paramsForInterRAHT.voxelCount; i++) {
      UrahtNode node_ref;
      node_ref.pos = attrInterPredParams.paramsForInterRAHT.mortonCode[i];
      node_ref.weight = 1;
      node_ref.qp = {0, 0};
      weightsLf_ref.emplace_back(node_ref);
      for (int k = 0; k < numAttrs; k++) {
        attrsLf_ref.push_back(attrInterPredParams.paramsForInterRAHT.attributes[i * numAttrs + k]);
      }
    }

    weightsHf_ref.reserve(attrInterPredParams.paramsForInterRAHT.voxelCount);
    attrsHf_ref.reserve(attrInterPredParams.paramsForInterRAHT.voxelCount * numAttrs);
  }
  
  // ascend tree
  std::vector<int> levelHfPos;
  std::vector<int> levelHfPos_ref;

  int numDupNodes = numPoints;

  for (int level = 0, numNodes = weightsLf.size(); numNodes > 1; level++) {
    levelHfPos.push_back(weightsHf.size());
    if (level == 0) {
      // process any duplicate points
      numNodes = reduceUnique_decoder(numNodes, &weightsLf, &weightsHf);
      numDupNodes -= numNodes;
    } else {
      // normal level reduction
      numNodes = reduceLevel_decoder(level, numNodes, &weightsLf, &weightsHf);
    }
  }
  
  if (enableACInterPred){
    for (int level = 0, numNodes = weightsLf_ref.size(); numNodes > 1; level++) {
      levelHfPos_ref.push_back(weightsHf_ref.size());
      if (level == 0) {
        // process any duplicate points
        numNodes = reduceUnique<haarFlag, numAttrs>(
          numNodes, &weightsLf_ref, &weightsHf_ref, &attrsLf_ref, &attrsHf_ref);
      } else {
        // normal level reduction
        numNodes = reduceLevel<haarFlag, numAttrs>(
          level, numNodes, &weightsLf_ref, &weightsHf_ref, &attrsLf_ref, &attrsHf_ref);
      }
    }    
  }

  assert(weightsLf[0].weight == numPoints);

  // reconstruction buffers
  std::vector<int64_t> attrRec, attrRecParent;
  attrRec.resize(numPoints * numAttrs);
  attrRecParent.resize(numPoints * numAttrs);

  std::vector<int64_t> attrRecUs, attrRecParentUs;
  attrRecUs.resize(numPoints * numAttrs);
  attrRecParentUs.resize(numPoints * numAttrs);

  std::vector<UrahtNode> weightsParent;
  weightsParent.reserve(numPoints);

  std::vector<int8_t> numParentNeigh, numGrandParentNeigh;
  numParentNeigh.resize(numPoints);
  numGrandParentNeigh.resize(numPoints);

  // quant layer selection
  auto qpLayer = 0;
  // AC coeff QP offset laer
  auto acCoeffQpLayer = -1;

  // descend tree
  weightsLf.resize(1);
  attrsLf.resize(numAttrs);

  weightsLf_ref.resize(1);
  attrsLf_ref.resize(numAttrs);

  int sumNodes = 0;
  int8_t LcpCoeff = 0;
  int filterIdx = 0;

  // ----------------------------------- descend tree, loop on level ------------------------------------
  for (int level = levelHfPos.size() - 1, level_ref = levelHfPos_ref.size() - 1, isFirst = 1; level > 0; /*nop*/) {
    int numNodes = weightsHf.size() - levelHfPos[level];
    sumNodes += numNodes;
    weightsLf.resize(weightsLf.size() + numNodes);
    attrsLf.resize(attrsLf.size() + numNodes * numAttrs);
    expandLevel_decoder(level, numNodes, &weightsLf, &weightsHf);

    weightsHf.resize(levelHfPos[level]);
    attrsHf.resize(levelHfPos[level] * numAttrs);

    if (level_ref <= 0) {
      enableACInterPred = false;
    }

    if (treeDepth >= treeDepthLimit)
      enableACInterPred = false;

    if (enableACInterPred) {
      int numNodes_ref = weightsHf_ref.size() - levelHfPos_ref[level_ref];
      weightsLf_ref.resize(weightsLf_ref.size() + numNodes_ref);
      attrsLf_ref.resize(attrsLf_ref.size() + numNodes_ref * numAttrs);

      expandLevel<haarFlag, numAttrs>(
        level_ref, numNodes_ref, &weightsLf_ref, &weightsHf_ref, &attrsLf_ref, &attrsHf_ref);

      weightsHf_ref.resize(levelHfPos_ref[level_ref]);
      attrsHf_ref.resize(levelHfPos_ref[level_ref] * numAttrs);
    }

    // expansion of level is complete, processing is now on the next level
    level--;
    level_ref--;

    // every three levels, perform transform
    if (level % 3)
      continue;
    //if current level nodes number is equal to previous nodes level, skip current level
    if (sumNodes == 0)
      continue;

    // initial scan position of the coefficient buffer
    //  -> first level = all coeffs
    //  -> otherwise = ac coeffs only
    bool inheritDc = !isFirst;
    bool enablePredictionInLvl = inheritDc && rahtPredParams.raht_prediction_enabled_flag;
    isFirst = 0;

	//--------------- initialize parameters of layer RDO for current level ------------
    enableACRDOInterPred =
      attrInterPredParams.paramsForInterRAHT.raht_enable_inter_intra_layer_RDO
      && enableACInterPred && enablePredictionInLvl;

	enableACRDONonPred = enablePredictionInLvl
      ? rahtPredParams.raht_enable_intraPred_nonPred_code_layer
      : false;

    const bool& enableRDOCodingLayer = enableACRDOInterPred || enableACRDONonPred;

	bool curLevelEnableACInterPred = false;
    bool curLevelEnableACIntraPred = false;
    int predMode = 0;
    if (enableRDOCodingLayer) {
      predMode = coder.decodeMode(enableACRDOInterPred, enableACRDONonPred);
      if (enableACRDOInterPred) {
        if (enableACRDONonPred) {  // 0: intraPred 1:interPred 2:nonPred
          curLevelEnableACIntraPred = predMode == 0;
          curLevelEnableACInterPred = predMode == 1;
        } else {  // 0: intraPred 1:interPred
          curLevelEnableACIntraPred = predMode == 0;
          curLevelEnableACInterPred = predMode == 1;
        }
      } else if (enableACRDONonPred) {
        curLevelEnableACIntraPred = predMode == 0;  // 0: intraPred 1:nonPred
      }
    }

	//--------------- initialize LCP coeff for current level ------------
    LcpCoeff = 0;
    PCCRAHTComputeLCP curlevelLcp;

	//--------------- calculate parent node information for current level ------------ 
    if (enablePredictionInLvl) {
      for (auto& ele : weightsParent)
        ele.occupancy = 0;

      const int parentCount = weightsParent.size();
      auto it = weightsLf.begin();
      for (auto i = 0; i < parentCount; i++) {
        weightsParent[i].firstChild = it++;

        while (it != weightsLf.end()
               && !((it->pos ^ weightsParent[i].pos) >> (level + 3)))
          it++;
        weightsParent[i].lastChild = it;
      }
    }   
    
    //--------------- select quantiser according to transform layer ------------
    qpLayer = std::min(qpLayer + 1, int(qpset.layers.size()) - 1);
    acCoeffQpLayer++;
   
    //--------------- prepare reconstruction buffers ------------
    //  previous reconstruction -> attrRecParent
    std::swap(attrRec, attrRecParent);
    std::swap(attrRecUs, attrRecParentUs);
    std::swap(numParentNeigh, numGrandParentNeigh);
    auto attrRecParentUsIt = attrRecParentUs.cbegin();
    auto attrRecParentIt = attrRecParent.cbegin();
    auto weightsParentIt = weightsParent.begin();
    auto numGrandParentNeighIt = numGrandParentNeigh.cbegin();
    
	//---------------- get inter filter of current level ---------------------
    int64_t interFilterTap = 128;
    int64_t quantizedResFilterTap = 0;

    if ((!enableFilterEstimation) && (enableACInterPred) && (treeDepth < treeDepthLimit)) {
      int filtexidx = treeDepth < fixedFilterTaps.size() ? treeDepth : (fixedFilterTaps.size()-1);
      interFilterTap = fixedFilterTaps[filtexidx];
    }    

    //get filter tap at the decoder 
    bool enableDecoderParsing = false;

    if (enableFilterEstimation && enableACRDOInterPred && curLevelEnableACInterPred && 
      (treeDepth >= skipInitLayersForFiltering))
    {
      enableDecoderParsing = true;
    }
    else if (enableFilterEstimation && !enableACRDOInterPred && enableACInterPred && 
      (treeDepth >= skipInitLayersForFiltering)) 
    {
      enableDecoderParsing = true;
    }

    if (enableDecoderParsing) {
      auto quantizers = qpset.quantizers(qpLayer, {0,0});
      auto& q = quantizers[0];
      quantizedResFilterTap = attrInterPredParams.paramsForInterRAHT.FilterTaps[filterIdx];
      filterIdx++;
      int64_t recResidueFilterTap = divExp2RoundHalfUp(q.scale(quantizedResFilterTap), kFixedPointAttributeShift);
      interFilterTap = 128 - recResidueFilterTap;
    }

	// ----------------------------- loop on nodes of current level -----------------------------------
    for (int i = 0, j = 0, iLast, jLast, iEnd = weightsLf.size(), jEnd = weightsLf_ref.size(); i < iEnd; i = iLast) {
      FixedPoint SampleBuf[6][8] = {0}, transformBuf[6][8] = {0};
      FixedPoint (*SamplePredBuf)[8] = &SampleBuf[numAttrs], (*transformPredBuf)[8] = &transformBuf[numAttrs];
      FixedPoint SampleInterPredBuf[3][8] = {0}, transformInterPredBuf[3][8] = {0};
      FixedPoint PredDC[3] = {0};
      FixedPoint NodeRecBuf[3][8] = {0};
      FixedPoint normalizedSqrtBuf[8] = {0};

	  // For Lcp prediction
	  int64_t CoeffRecBuf[8][3] = {0};     
      FixedPoint transformResidueRecBuf[3] = {0};

      int weights[8 + 8 + 8 + 8] = {};
      uint64_t sumWeights_ref = 0, sumWeights_cur = 0; 
	  FixedPoint finterDC[3] = {0}, interParentMean[3] = {0};
      uint8_t occupancy = 0;
      int nodelvlSum = 0;
      Qps nodeQp[8] = {};
      // generate weights, occupancy mask, and fwd transform buffers
      // for all siblings of the current node.
      int nodeCnt = 0;

      int weights_ref[8 + 8 + 8 + 8] = {};
      bool interNode = false;

      bool checkInterNode = enableACInterPred;
      if(enableACRDOInterPred)
        checkInterNode = curLevelEnableACInterPred;

      if (checkInterNode) {
        const auto cur_pos = weightsLf[i].pos >> (level + 3);
        auto ref_pos = weightsLf_ref[j].pos >> (level_ref + 3);
        while ((j < weightsLf_ref.size() - 1) && (cur_pos > ref_pos)) {
          j++;
          ref_pos = weightsLf_ref[j].pos >> (level_ref + 3);
        }
        if (cur_pos == ref_pos) {
          interNode = true;
        }
      }

      if (interNode) {
        for (jLast = j; jLast < jEnd; jLast++) {
          int nextNode = jLast > j
          && !isSibling(weightsLf_ref[jLast].pos, weightsLf_ref[j].pos, level_ref + 3);
          if (nextNode)
            break;
          int nodeIdx = (weightsLf_ref[jLast].pos >> level_ref) & 0x7;
          weights_ref[nodeIdx] = weightsLf_ref[jLast].weight;
          sumWeights_ref += (uint64_t) weights_ref[nodeIdx];
          for (int k = 0; k < numAttrs; k++){
            SampleInterPredBuf[k][nodeIdx] = attrsLf_ref[jLast * numAttrs + k];
            finterDC[k] += SampleInterPredBuf[k][nodeIdx];
          }
        }
        
        if(haarFlag){
          mkWeightTree(weights_ref);
          std::copy_n(&SampleInterPredBuf[0][0], numAttrs * 8, &transformInterPredBuf[0][0]);
          ComputeDCfor222Block<HaarKernel>(numAttrs, transformInterPredBuf, weights_ref);
          for (int k = 0; k < numAttrs; k++) {
            finterDC[k].val = transformInterPredBuf[k][0].val;
            interParentMean[k].val = finterDC[k].val;
          }
        }
        else{
          FixedPoint rsqrtWeightSumRef(0);
          int shiftBits = sumWeights_ref > 1024 ? ilog2(sumWeights_ref - 1) >> 1 : 0;
          rsqrtWeightSumRef.val = irsqrt(sumWeights_ref) >> (40 - shiftBits - FixedPoint::kFracBits);
          for (int k = 0; k < numAttrs; k++) {
            finterDC[k].val >>= shiftBits;
            finterDC[k] *= rsqrtWeightSumRef;
            interParentMean[k].val = finterDC[k].val;
            interParentMean[k].val >>= shiftBits;
            interParentMean[k] *= rsqrtWeightSumRef;
          }
        }
        
        int64_t curinheritDC = (inheritDc) ? *attrRecParentUsIt : 0;
        int64_t interDC = finterDC[0].val;
        
        if ((curinheritDC > 0) && (interDC > 0) && (!haarFlag)) {
          bool condition1 = 10 * interDC < ((curinheritDC)* 5);
          bool condition2 = 10 * interDC > ((curinheritDC)* 20);
          if (condition1 || condition2) {
            interNode = false;
          }
        }
      }

      for (iLast = i; iLast < iEnd; iLast++) {
        int nextNode = iLast > i
          && !isSibling(weightsLf[iLast].pos, weightsLf[i].pos, level + 3);
        if (nextNode)
          break;

        int nodeIdx = (weightsLf[iLast].pos >> level) & 0x7;
        weights[nodeIdx] = weightsLf[iLast].weight;
        sumWeights_cur += (uint64_t) weights[nodeIdx];
        nodeQp[nodeIdx][0] = weightsLf[iLast].qp[0] >> regionQpShift;
        nodeQp[nodeIdx][1] = weightsLf[iLast].qp[1] >> regionQpShift;

        occupancy |= 1 << nodeIdx;

        if (rahtExtension)
          nodeCnt++;

      }

      if (!inheritDc) {
        for (int j = i, nodeIdx = 0; nodeIdx < 8; nodeIdx++) {
          if (!weights[nodeIdx])
            continue;
          numParentNeigh[j++] = 19;
        }
      }

      if (rahtExtension && nodeCnt == 1){
        interNode = false;
      }

      mkWeightTree(weights);

      // Inter-level prediction:
      //  - Find the parent neighbours of the current node
      //  - Generate prediction for all attributes into transformPredBuf
      //  - Subtract transformed coefficients from forward transform
      //  - The transformPredBuf is then used for reconstruction
      bool enablePrediction = enablePredictionInLvl;

      if (enableACRDONonPred)
        enablePrediction = curLevelEnableACIntraPred;

      if (enableACInterPred) {
        if (curLevelEnableACInterPred && !interNode)
          enablePrediction = enablePredictionInLvl;
        else if (curLevelEnableACIntraPred)
          enablePrediction = enablePredictionInLvl;
      }

      bool eligiblePrediction = enablePrediction;
          
      if (enablePredictionInLvl) {
        weightsParentIt->occupancy = occupancy;
        // indexes of the neighbouring parents
        int parentNeighIdx[19];
        int childNeighIdx[12][8];

        int parentNeighCount = 0;
        if (rahtExtension && nodeCnt == 1) {
          enablePrediction = false;
          eligiblePrediction = false;
          parentNeighCount = 19;
        } else if (
          *numGrandParentNeighIt < rahtPredParams.raht_prediction_threshold0) {
          enablePrediction = false;
          eligiblePrediction = false;
        } else {
          findNeighbours(
            weightsParent.begin(), weightsParent.end(), weightsParentIt,
            weightsLf.begin(), weightsLf.begin() + i, level + 3, occupancy,
            parentNeighIdx, childNeighIdx,
            rahtPredParams.raht_subnode_prediction_enabled_flag, rahtPredParams.raht_prediction_search_range);
          for (int i = 0; i < 19; i++) {
            parentNeighCount += (parentNeighIdx[i] != -1);
          }
          if (parentNeighCount < rahtPredParams.raht_prediction_threshold1) {
            enablePrediction = false;
          } 
        }

        if (enableACRDONonPred)
          enablePrediction = enablePrediction || eligiblePrediction;

        if (enablePrediction) {
          int64_t limitLow = 0;
          int64_t limitHigh = 0;
          intraDcPred_decoder<haarFlag, numAttrs, rahtExtension>(
            parentNeighIdx, childNeighIdx, occupancy,
            attrRecParent.begin(), attrRec.begin(),
            SamplePredBuf, rahtPredParams, limitLow, limitHigh);
        }

        for (int j = i, nodeIdx = 0; nodeIdx < 8; nodeIdx++) {
          if (!weights[nodeIdx])
            continue;
          numParentNeigh[j++] = parentNeighCount;
        }
      }

      int parentWeight = 0;
      if (inheritDc) {
        parentWeight = weightsParentIt->weight;
        weightsParentIt++;
        numGrandParentNeighIt++;
      }

      bool enableIntraPrediction =
      curLevelEnableACInterPred && enablePrediction;
      bool enableInterPrediction = curLevelEnableACInterPred;
      
      if(haarFlag){
        if(interNode){
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            if(weights[childIdx] == 0){
              for(int k = 0; k < numAttrs; k++){
                SampleInterPredBuf[k][childIdx].val = 0;
              }
              continue;
            } else if(weights_ref[childIdx] == 0){
              for(int k = 0; k < numAttrs; k++){
                SampleInterPredBuf[k][childIdx].val = interParentMean[k].val;
              }
            }
            for(int k = 0; k < numAttrs; k++)
              SampleInterPredBuf[k][childIdx].val = (SampleInterPredBuf[k][childIdx].val >> FixedPoint::kFracBits) << (FixedPoint::kFracBits);
          }
          enablePrediction = true;
          std::copy_n(&SampleInterPredBuf[0][0], numAttrs * 8, &SamplePredBuf[0][0]);
        }
      }
      else {
        // normalise coefficients
        if (interNode){
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            if(weights[childIdx] == 0){
              for(int k = 0; k < numAttrs; k++){
                SampleInterPredBuf[k][childIdx].val = 0;
              }
              continue;
            } else if(weights_ref[childIdx] == 0){
              for(int k = 0; k < numAttrs; k++){
                SampleInterPredBuf[k][childIdx].val = interParentMean[k].val;
              }
            }
            if(weights_ref[childIdx]>1) {
              FixedPoint rsqrtWeight;
              uint64_t w = weights_ref[childIdx];
              int shift = w > 1024 ? ilog2(w - 1) >> 1 : 0;
              rsqrtWeight.val = irsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
              for (int k = 0; k < numAttrs; k++) {
                SampleInterPredBuf[k][childIdx].val >>= shift;
                SampleInterPredBuf[k][childIdx] *= rsqrtWeight; //sqrt normalized: DC
                SampleInterPredBuf[k][childIdx].val >>= shift;
                SampleInterPredBuf[k][childIdx] *= rsqrtWeight; //mean attribute
              }
            }
          }
          enablePrediction = true;
          std::copy_n(&SampleInterPredBuf[0][0], numAttrs * 8, &SamplePredBuf[0][0]);
        }
        
        for (int childIdx = 0; childIdx < 8; childIdx++) {
          
          if (weights[childIdx] <= 1)
            continue;
          
          // Predicted attribute values
          FixedPoint sqrtWeight;
          if (enablePrediction) {
            sqrtWeight.val =
            isqrt(uint64_t(weights[childIdx]) << (2 * FixedPoint::kFracBits));
            for (int k = 0; k < numAttrs; k++) {
              SamplePredBuf[k][childIdx] *= sqrtWeight;
            }
          }
        }
      }

      // forward transform:
      //  - decoder: just transform prediction
      if (haarFlag) {
        if (enablePrediction){
          std::copy_n(&SamplePredBuf[0][0], numAttrs * 8, &transformPredBuf[0][0]);
          fwdTransformBlock222<HaarKernel>(numAttrs, transformPredBuf, weights);
        }    
      }
      else {      
        if(interNode) //temporal filtering
        {
          for (int childIdx = 0; childIdx < 8; childIdx++) {
            for (int k = 0; k < numAttrs; k++){
              int64_t refVal = 0, filteredVal = 0;
              refVal = SamplePredBuf[k][childIdx].val;
              filteredVal = (treeDepth < skipInitLayersForFiltering) ? refVal
			    : (refVal * interFilterTap ) >> 7;
              SamplePredBuf[k][childIdx].val = filteredVal;
            }
          }
        }
      }
        
      //compute DC of the predictions: Done in the same way at the encoder and decoder to avoid drifting
      if(enablePrediction && !haarFlag){
        FixedPoint rsqrtweightsum;
        rsqrtweightsum.val = irsqrt(sumWeights_cur);
        for (int childIdx = 0; childIdx < 8; childIdx++) {
          if (weights[childIdx] == 0)
            continue;
          FixedPoint normalizedsqrtweight;
          if (weights[childIdx] == 1) {
            normalizedsqrtweight.val = rsqrtweightsum.val >> (40 - FixedPoint::kFracBits);
          } else {
            FixedPoint sqrtWeight;
            sqrtWeight.val = isqrt(uint64_t(weights[childIdx]) << (2 * FixedPoint::kFracBits));;
            normalizedsqrtweight.val = sqrtWeight.val * rsqrtweightsum.val >> 40;
          }
          normalizedSqrtBuf[childIdx] = normalizedsqrtweight;
          for (int k = 0; k < numAttrs; k++){
            FixedPoint prod;
            prod.val = normalizedsqrtweight.val; prod *= SamplePredBuf[k][childIdx];
            PredDC[k].val += prod.val;           
          }
        }
      }
      
      //flags for skiptransform
      bool skipTransform = enablePrediction;
      
      // per-coefficient operations:
      //  - subtract transform domain prediction (encoder)
      //  - subtract the prediction between chroma channel components
      //  - write out/read in quantised coefficients
      //  - inverse quantise + add transform domain prediction
      scanBlock(weights, [&](int idx) {
        // skip the DC coefficient unless at the root of the tree
        if (inheritDc && !idx)
          return;

        // Check if QP offset is to be applied to AC coeffiecients
        Qps coeffQPOffset = (acCoeffQpLayer <= maxAcCoeffQpOffsetLayers && idx)
          ? qpset.rahtAcCoeffQps[acCoeffQpLayer][idx - 1]
          : Qps({0, 0});

        Qps nodeQPOffset = {
          nodeQp[idx][0] + coeffQPOffset[0],
          nodeQp[idx][1] + coeffQPOffset[1]};

        // The RAHT transform
        auto quantizers = qpset.quantizers(qpLayer, nodeQPOffset);
        for (int k = 0; k < numAttrs; k++) {

          auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];

          int64_t coeff = *coeffBufItK[k]++;
          transformResidueRecBuf[k] = CoeffRecBuf[nodelvlSum][k] =
            divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);

          if (!haarFlag && enableLCPPred) {
            if (k != 2) {
              NodeRecBuf[k][idx] = transformResidueRecBuf[k];
            } 
			else if (k == 2) {
              transformResidueRecBuf[k].val +=
                (LcpCoeff * transformResidueRecBuf[1].val) >> 4;
              NodeRecBuf[k][idx] = transformResidueRecBuf[k];
              CoeffRecBuf[nodelvlSum][k] = transformResidueRecBuf[k].round();
            }
          } 
		  else {
            transformPredBuf[k][idx] += transformResidueRecBuf[k];
            NodeRecBuf[k][idx] = transformResidueRecBuf[k];
          }
          skipTransform = skipTransform && (NodeRecBuf[k][idx].val == 0);          
        }
        nodelvlSum++;
      });

      // compute last component coefficient
      if (numAttrs == 3 && nodeCnt > 1 && !haarFlag && inheritDc && enableLCPPred) {
        LcpCoeff = curlevelLcp.computeLCPCoeff(nodelvlSum, CoeffRecBuf);
      }
      
      if(haarFlag){
        std::copy_n(&transformPredBuf[0][0], numAttrs * 8, &NodeRecBuf[0][0]);   
      }
      // replace DC coefficient with parent if inheritable
      if (inheritDc) {
        for (int k = 0; k < numAttrs; k++) {
          attrRecParentIt++;
          int64_t val = *attrRecParentUsIt++;
          if (rahtExtension){
            NodeRecBuf[k][0].val = (haarFlag)? val: val - PredDC[k].val;
          }
          else if (val > 0)
            transformPredBuf[k][0].val = val << (15 - 2);
          else
            transformPredBuf[k][0].val = -((-val) << (15 - 2));
        }
      }

      if (haarFlag) {
        invTransformBlock222<HaarKernel>(numAttrs, NodeRecBuf, weights);
      } else {
        // apply skip transform here
        if (skipTransform) {
          FixedPoint DCerror[3];
          for (int k = 0; k < numAttrs; k++) {
            DCerror[k] = NodeRecBuf[k][0]; NodeRecBuf[k][0].val = 0;
          }
          for (int cidx = 0; cidx < 8; cidx++) {
            if (!weights[cidx])
              continue;
            
            for(int k = 0; k < numAttrs; k++) {
              FixedPoint Correctionterm = normalizedSqrtBuf[cidx];  Correctionterm *= DCerror[k];
              NodeRecBuf[k][cidx] = Correctionterm;
            }
          }  
        } 
        else{
          invTransformBlock222<RahtKernel>(numAttrs, NodeRecBuf, weights);
        }
      }

      for (int j = i, nodeIdx = 0; nodeIdx < 8; nodeIdx++) {
        if (!weights[nodeIdx])
          continue;

        for (int k = 0; k < numAttrs; k++)
          if (rahtExtension) {
            if(!haarFlag){
              NodeRecBuf[k][nodeIdx].val += SamplePredBuf[k][nodeIdx].val;
            }
            attrRecUs[j * numAttrs + k] = NodeRecBuf[k][nodeIdx].val;            
          }
          else {
            FixedPoint temp = transformPredBuf[k][nodeIdx];
            temp.val <<= 2;
            attrRecUs[j * numAttrs + k] = temp.round();
          }

        // scale values for next level
        if (!haarFlag) {
          if (weights[nodeIdx] > 1) {
            FixedPoint rsqrtWeight;
            uint64_t w = weights[nodeIdx];
            int shift = w > 1024 ? ilog2(w - 1) >> 1 : 0;
            rsqrtWeight.val = irsqrt(w) >> (40 - shift - FixedPoint::kFracBits);
            for (int k = 0; k < numAttrs; k++) {
              NodeRecBuf[k][nodeIdx].val >>= shift;
              NodeRecBuf[k][nodeIdx] *= rsqrtWeight;
            }
          }
        }

        for (int k = 0; k < numAttrs; k++) {
          attrRec[j * numAttrs + k] = rahtExtension
              ? NodeRecBuf[k][nodeIdx].val
            : NodeRecBuf[k][nodeIdx].round();
        }
        j++;
      }
      // increment reference buffer index if inter prediction is enabled
    }//end loop on nodes of current level

    sumNodes = 0;
    // preserve current weights/positions for later search
    weightsParent = weightsLf;
    // increment tree depth
    treeDepth++;
  }//end loop on level

  // -------------- process duplicate points at level 0 if exists --------------
  if (numDupNodes) {
    std::swap(attrRec, attrRecParent);
    auto attrRecParentIt = attrRecParent.cbegin();
    auto attrsHfIt = attrsHf.cbegin();

    for (int i = 0, out = 0, iEnd = weightsLf.size(); i < iEnd; i++) {
      int weight = weightsLf[i].weight;
      // unique points have weight = 1
      if (weight == 1) {
        for (int k = 0; k < numAttrs; k++)
          attrRec[out++] = *attrRecParentIt++;
        continue;
      }
      Qps nodeQp = {
        weightsLf[i].qp[0] >> regionQpShift,
        weightsLf[i].qp[1] >> regionQpShift};

      // duplicates
      FixedPoint attrSum[3];
      FixedPoint attrRecDc[3];
      FixedPoint sqrtWeight;
      sqrtWeight.val = isqrt(uint64_t(weight) << (2 * FixedPoint::kFracBits));

      int64_t sumCoeff = 0;
      for (int k = 0; k < numAttrs; k++) {
        if (rahtExtension)
          attrRecDc[k].val = *attrRecParentIt++;
        else
          attrRecDc[k] = *attrRecParentIt++;
        if (!haarFlag) {
          attrRecDc[k] *= sqrtWeight;
        }
      }

      FixedPoint rsqrtWeight;
      for (int w = weight - 1; w > 0; w--) {
        RahtKernel kernel(w, 1);
        HaarKernel haarkernel(w, 1);
        int shift = w > 1024 ? ilog2(uint32_t(w - 1)) >> 1 : 0;

        auto quantizers = qpset.quantizers(qpLayer, nodeQp);
        for (int k = 0; k < numAttrs; k++) {
          auto& q = quantizers[std::min(k, int(quantizers.size()) - 1)];

          FixedPoint transformBuf[2];

          int64_t coeff = *coeffBufItK[k]++;
          transformBuf[1] =
            divExp2RoundHalfUp(q.scale(coeff), kFixedPointAttributeShift);
    
          // inherit the DC value
          transformBuf[0] = attrRecDc[k];

          if (haarFlag) {
            haarkernel.invTransform(
              transformBuf[0], transformBuf[1], &transformBuf[0],
              &transformBuf[1]);
          } else {
            kernel.invTransform(
              transformBuf[0], transformBuf[1], &transformBuf[0],
              &transformBuf[1]);
          }

          attrRecDc[k] = transformBuf[0];
          attrRec[out + w * numAttrs + k] =
            rahtExtension ? transformBuf[1].val : transformBuf[1].round();
          if (w == 1)
            attrRec[out + k] =
              rahtExtension ? transformBuf[0].val : transformBuf[0].round();
        }
      }

      attrsHfIt += (weight - 1) * numAttrs;
      out += weight * numAttrs;
    }
  }
  

  // -------------- write-back reconstructed attributes --------------
  assert(attrRec.size() == numAttrs * numPoints);
  if(rahtExtension)
    for (auto& attr : attrRec) {
      attr += FixedPoint::kOneHalf;
      *(attributes++) = attr >> FixedPoint::kFracBits;
    }
  else
    std::copy(attrRec.begin(), attrRec.end(), attributes);
}

//============================================================================
/*
 * inverse RAHT Fixed Point
 *
 * Inputs:
 * quantStepSizeLuma = Quantization step
 * mortonCode = list of 'voxelCount' Morton codes of voxels, sorted in ascending Morton code order
 * attribCount = number of attributes (e.g., 3 if attributes are red, green, blue)
 * voxelCount = number of voxels
 * coefficients = quantized transformed attributes array, in column-major order
 *
 * Outputs:
 * attributes = 'voxelCount' x 'attribCount' array of attributes, in row-major order
 *
 * Note output weights are typically used only for the purpose of
 * sorting or bucketing for entropy coding.
 */
void
regionAdaptiveHierarchicalInverseTransform(
  const RahtPredictionParams &rahtPredParams,
  const QpSet& qpset,
  const Qps* pointQpOffsets,
  int64_t* mortonCode,
  int* attributes,
  const int attribCount,
  const int voxelCount,
  int* coefficients,
  const bool rahtExtension,
  AttributeInterPredParams& attrInterPredParams,
  ModeDecoder& decoder)
{
  switch (attribCount) {
  case 3:
    if (rahtPredParams.integer_haar_enable_flag) {
      if (rahtExtension)
        uraht_process_decoder<true, 3, true>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<true, 3, false>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    } else {
      if (rahtExtension)
        uraht_process_decoder<false, 3, true>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<false, 3, false>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    }
    break;

  case 1:
    if (rahtPredParams.integer_haar_enable_flag) {
      if (rahtExtension)
        uraht_process_decoder<true, 1, true>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<true, 1, false>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    } else {
      if (rahtExtension)
        uraht_process_decoder<false, 1, true>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
      else
        uraht_process_decoder<false, 1, false>(
          rahtPredParams, qpset, pointQpOffsets, voxelCount, mortonCode,
          attributes, coefficients, attrInterPredParams, decoder);
    }
    break;
  default: throw std::runtime_error("attribCount only support 1 or 3");
  }
}

//============================================================================

}  // namespace pcc
