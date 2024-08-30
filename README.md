# A Comprehensive Evaluation of Arbitrary Image Style Transfer Methods

<div align=center>
<img src=https://github.com/ZhouZJ-DL/A-Comprehensive-Evaluation-of-Arbitrary-Image-Style-Transfer-Methods/blob/main/imgs/system.drawio2.png width=80% />
</div>


## 🚀 Overview
This research introduces a comprehensive assessment system for Arbitrary Image Style Transfer (AST) methods. Recognizing the limitations of existing evaluation approaches, our framework combines subjective user studies with objective metrics to provide a multi-grained understanding of AST performance. We collect a fine-grained dataset considering a range of image contexts such as different scenes, object complexities, and rich parsing information from multiple sources. Objective and subjective studies are conducted using the collected dataset.



## 🏁 ToDo
🗹 Release the user study website.<br>
🗹 Release the standard dataset with annotatio.<br>
☐ Release the code for extracting the ADE20K dataset to obtain image segmentation information.<br>
☐ Release the explanation and starter code for the standard dataset.<br>
☐ Release the evaluation code for AST methods.<br>



## 📲 Subjective Study
For our subjective study, we have created an interactive website where participants can engage. Welcome to our platform, designed to host a variety of user studies. We invite you to explore and participate at your convenience, and we hope you have a delightful experience! 🐶

*[Click here for user study](http://ivc.ia.ac.cn/)* 😺



## 📑 Requirements

### ADE20K Dataset: 
The ADE20K dataset comprises fully annotated images containing a diverse range of objects spanning over 3,000 object categories. These images also include detailed annotations for object parts. Additionally, ADE20K provides the original annotated polygons and object instances.

Given its comprehensive annotations and suitability for semantic understanding, we have selected ADE20K as an integral part of our collected dataset for developing the AST assessment. 
To utilize ADE20K, you can refer to their *[official repository](https://github.com/CSAILVision/ADE20K)*, including its download link and an introductory overview. 

We offer starter code that explores ADE20K dataset. Our provided code extracts the ADE20K dataset and parses it to extract essential features from each image, such as scene context, object complexity, and salient regions.

### WikiArt
We incorporate images sourced from *[WikiArt](https://www.wikiart.org/)* as a component of our collected dataset. *[WikiArt](https://www.wikiart.org/)* offers a diverse array of authentic artistic images, spanning various styles and genres. 

The Hugging Face dataset “*[wikiart](https://huggingface.co/datasets/huggan/wikiart)*” comprises paintings sourced from various artists and extracted from WikiArt. Each image in this dataset is accompanied by class labels. Leveraging this dataset, we can systematically evaluate how different artistic styles influence the stylization of images.



## 💿 Collected Standard Dataset
For the objective study, we have constructed a standard dataset including the following components:
- Content Images: These images are extracted from the ADE20K dataset.
- Style images: Sourced from WikiArt.
- Stylized Images: We generated the stylized images using 10 distinct AST methods.

<div align="center">

| AST method | type | Official Repository|
| :--------: | :--: | :-------: |
| ArtFusion | Diffusion-based |
| StyTr2 | Transformer-based |
| ArtFlow | Flow model |
| UCAST | CNN-based, contrast learning |
| MAST | CNN-based, manifold-based |
| SANet | CNN-based, attention-based |
| AdaIN | CNN-based |
| LST | CNN-based |
| NST | CNN-based |
| WCT | CNN-based |

</div>

Our collected dataset:
*[OneDrive](https://1drv.ms/u/c/de3ad968021b913b/ERwGC0OlZsZCn4Ur82kwiXABrcXac_8zSB1F1q_9IAkCKw?e=Ezp3XO)    [iCloud Drive](https://www.icloud.com/iclouddrive/0662WC1QtpRCa0g0jZDa4ItNw#custom%5Fdataset)* 🐳
