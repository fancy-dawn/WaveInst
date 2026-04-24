# WaveInst: A Frequency-Domain Enhanced Network for Fine-Grained Thin Tree Trunk Extraction in Forest Scenes

## Overview

WaveInst consists of four main components: the backbone, the encoder, the decoder, and a Frequency-domain Feature Compensation (FFC) branch. The FFC branch is composed of a Discrete Wavelet Transformation (DWT) Block and a High-Frequency Enhancement (HFE) Block. The encoder aggregates multi-scale features extracted by the backbone through a Feature Pyramid Network (FPN) and utilizes the proposed Adaptive Gated Fusion Module (AGFM) to adaptively fuse semantic and frequency-domain information. The decoder employs CoordConv and a decoupled dual-branch design: the DRMask branch reconstructs fine spatial details and generates segmentation masks, while the Inst branch discriminates instances and refines instance-level predictions. Finally, a bipartite matching mechanism ensures precise supervision for accurate instance segmentation.

<center>
<img src="./assets/waveinst.png">
</center>

## Installation and Prerequisites

## Pre-trained weight on SynthTree43k

## Acknowledgements

WaveInst is based on [SparseInst](https://github.com/hustvl/SparseInst), [mmdetection](https://github.com/open-mmlab/mmdetection), and we sincerely thanks for their code and contribution to the community!

## License

WaveInst is released under the [MIT Licence](LICENCE).