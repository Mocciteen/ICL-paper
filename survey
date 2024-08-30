## Related Work About ICL

### 1. In-Context Learning

​	In-context learning (ICL) has been pointed out firstly by **Brown et al.2020**[^1].They  argued that by giving a frozen GPT3 model some additional examples related to test samples, the reasoning result of LLM can be significantly improved. Later, many efforts have been made on in-context learning (ICL).

​	Some analytical studies have discussed the underlying mechanism of ICL. **J. Oswald et al.2022** [^2]; **Dai et al.2022**[^3]argued that ICL capability is closely related to transformer architecture. Meanwhile **Xie et al.2021** [^4]tried to explain ICL from the Angle of implicit bayesian inference. In addition, **Min et al.2022** [^5] analyzed what factors affect ICL. 

​	Compared with the traditional **supervised fine-tuning methods(Ding et al.2023[^6]),** which requires a training stage that uses backward gradients to update model parameters, ICL does not need to adjust the parameters of the model and only uses several examples related to the task to learn by analogy**(Winston.1980[^7])**. So, as a new paradigm, the advantage of ICL is that it can quickly adapt to new tasks and can greatly reduce the computational costs.

​	Although ICL can improve the reasoning skills of LLM, this capability is sensitive to demonstration designing**(Zhao et al.2021[^8])**, some ongoing research on ICL has also explored  the impact of ($i$) demonstration selection**(Liu et al.2022[^9]; Levy et al.2023[^10]; Gupta et al.2023[^11]; Rubin et al.2022[^12])** ($ii$) demonstration ordering**(Zhao et al.2021[^13];Liu et al.2021[^14];Lu et al.2021[^15])** ($iii$) demonstration formatting**(Mishra et al.2021[^16];Wang et al.2022[^17])**  on ICL capabilities.



### 2. Example Selection in ICL

​	One of the central issues in ICL is the selection of examples. The most direct way to select examples about the test input is to select the top-k examples with scores based on the correlation measure. Classic methods include the use of cosine similarity  and **BM25(Trotman et al.2014[^18])**. While cosine similarity captures some aspects of semantic similarity, it is limited to a single embedding. A better way to capture salient aspects is to use contextualized token embeddings pointed out  by **Zhang et al.2019[^19]** ,which called BERTScore.

​	In addition to the unsupervised selection methods described above, some studies have also used methods of training retrievers to select examples**(Rubin et al.2021[^20])**. But this type of supervised learning would compromise the ICL's advantage of not needing to train complex parameters

​	Besides, these studies considered each example separately, which often leaded to a lack of diversity and may not result in the best combination of k examples, as these examples may be homogeneous. So, **Levy et al.2022[^21]; Gupta et al.2023[^22]** took coverage into account to reduce redundancy; **Ye et al.2023 [^23]** used Determinantal Point Processes **(Kulesza et al.2012)[^24]** to select a diverse set of demonstrations similar to the test instance. Besides, **Qin et al.2023[^25]** increased diversity by using different reasoning paths through iterative training. **Mo et al.2024[^26]**suggested that it is necessary to exploit the value in negative examples, thus enabling more comprehensive information extraction capabilities.

​	In order to further increase the reliability of the ICL, the inference process is also taken into account within the context. **Qin et al.2023[^27]** introduced a zero-shot-CoT-based approach to minimize misleading information in ICL examples.

​	Other recent studies have used graph-based approaches to solve this pronlem, *e.g.* using KNN method **(Liu et al.2021[^28])**; selecing examples based on the influence of a collection in a social network **(Zhang et al.2023[^29])** ; considering the influence of a certain point on the set coverage based on the idea of clustering **(Mavromatis et al.2023[^30])**. Because graphs have a good set of properties and the graph-based approach has the advantage of good visualization, I considered using a graph-related approach to incorporate similarity and diversity into the example selection considerations.



### 3. ICL-oriented Prompt Compression

​	Recently,  researchers attempted to utilize soft prompts to convert actual tokens to denseinformation virtual tokens.  **Wingate et al.2022[^31]** proposed a method to learn compact soft prompts to simulate the original natural prompt by optimizing the KL divergence. To improve compression, **Chevalier et al.2023 [^32]**proposed AutoCompressor, which takes a compression cumulation approach to generate compressed virtual tokens. However, AutoCompressor breaks the independence of the examples and is computatively complex. **Ge et al.2023[^33]** proposed an ICAE using Adapter Tuning method to compress the processed presentation into virtual tokens. **Mu et al.2023[^34]** used the GIST method of adding attention mask, but GIST needs to be fine-tuned to the LLM, and the obtained gist tokens also need to be used in the specially tuned LLM; **Gao et al.2024[^35]** proposed a UniICL method to unify the process of context compression, selection and inference.

​	However, the existing work neglects the function of the surface information of sentences when extracting abstract information by compression. Inspired by the extraction of information at different levels of images in image processing, I think a more perfect method is to combine the surface information of sentences with the abstract information, and we can further consider the integration of reasoning paths into the information agent of sentences. This can make the information held by the sentence encoded more comprehensive and comprehensive.



## 4.Preliminaries

​	LLMs have the ability to solve new task easily when prompted with a few examples of that task.

​	Given a training set  $$D = {(xi, yi)}_i^{n=1}$$of input-output pairs, and a test inut $$x_{test}$$, ICL means that LLM can generate the following test output with conditionally using the training set $$D$$:
$$
y_{test} ∼ P_{LLM} (· | (x_1, y_1), . . . , (x_n, y_n), x_{test} ))
$$
​	And the goal of ICL demonstration selection is to seek a easy and efficient way that can find a subset of training examples$$P = {(x_j, y_j)}_j^{m=1}$$from $$D$$.through choosing the most relevant subset of candidates. So, ICL can improvethe computational efficiency and performance of LLM, which means that this subset will maximizes the probability of generating the desired $$y_{test}$$ when the Inference LLM is conditioned on $$x_{test}$$ and $$P$$.



## Reference

[^1]: Brown, Tom B. et al. “Language Models are Few-Shot Learners.” *ArXiv* abs/2005.14165 (2020): n. pag.
[^2]: Oswald, Johannes von et al. “Transformers learn in-context by gradient descent.” *International Conference on Machine Learning* (2022).
[^3]: Dai, Damai et al. “Why Can GPT Learn In-Context? Language Models Implicitly Perform Gradient Descent as Meta-Optimizers.” (2022).
[^4]: Xie, Sang Michael et al. “An Explanation of In-context Learning as Implicit Bayesian Inference.” *ArXiv* abs/2111.02080 (2021): n. pag.
[^5]: Min, Sewon et al. “Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?” *ArXiv* abs/2202.12837 (2022): n. pag.
[^6]: Ding, Ning et al. “Parameter-efficient fine-tuning of large-scale pre-trained language models.” *Nature Machine Intelligence* 5 (2023): 220-235.
[^7]: Winston, Patrick Henry. “Learning and reasoning by analogy.” *Commun. ACM* 23 (1980): 689-703.
[^8]: Zhao, Tony et al. “Calibrate Before Use: Improving Few-Shot Performance of Language Models.” *International Conference on Machine Learning* (2021).
[^9]: Liu, Jiachang et al. “What Makes Good In-Context Examples for GPT-3?” *Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out* (2021).
[^10]: Levy, Itay et al. “Diverse Demonstrations Improve In-context Compositional Generalization.” *ArXiv* abs/2212.06800 (2022): n. pag.
[^11]: Gupta, Shivanshu et al. “Coverage-based Example Selection for In-Context Learning.” *ArXiv* abs/2305.14907 (2023): n. pag.
[^12]: Rubin, Ohad et al. “Learning To Retrieve Prompts for In-Context Learning.” *ArXiv* abs/2112.08633 (2021): n. pag.
[^13]: Zhao, Tony et al. “Calibrate Before Use: Improving Few-Shot Performance of Language Models.” *International Conference on Machine Learning* (2021).
[^14]: Liu, Jiachang et al. “What Makes Good In-Context Examples for GPT-3?” *Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out* (2021).
[^15]: Lu, Yao et al. “Fantastically Ordered Prompts and Where to Find Them: Overcoming Few-Shot Prompt Order Sensitivity.” *ArXiv* abs/2104.08786 (2021): n. pag.
[^16]: Mishra, Swaroop et al. “Cross-Task Generalization via Natural Language Crowdsourcing Instructions.” *Annual Meeting of the Association for Computational Linguistics* (2021).
[^17]: Wang, Yizhong et al. “Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ NLP Tasks.” *Conference on Empirical Methods in Natural Language Processing* (2022).
[^18]: Trotman, Andrew et al. “Improvements to BM25 and Language Models Examined.” *Proceedings of the 19th Australasian Document Computing Symposium* (2014): n. pag.
[^19]: Zhang, Tianyi et al. “BERTScore: Evaluating Text Generation with BERT.” *ArXiv* abs/1904.09675 (2019): n. pag.
[^20]: Rubin, Ohad et al. “Learning To Retrieve Prompts for In-Context Learning.” *ArXiv* abs/2112.08633 (2021): n. pag.
[^21]: Levy, Itay et al. “Diverse Demonstrations Improve In-context Compositional Generalization.” *ArXiv* abs/2212.06800 (2022): n. pag.
[^22]: Gupta, Shivanshu et al. “Coverage-based Example Selection for In-Context Learning.” *ArXiv* abs/2305.14907 (2023): n. pag.
[^23]: Ye, Jiacheng et al. “Compositional Exemplars for In-context Learning.” *International Conference on Machine Learning* (2023).
[^24]: Kulesza, Alex and Ben Taskar. “Determinantal Point Processes for Machine Learning.” *Found. Trends Mach. Learn.* 5 (2012): 123-286.
[^25]: Qin, Chengwei et al. “In-Context Learning with Iterative Demonstration Selection.” *ArXiv* abs/2310.09881 (2023): n. pag.
[^26]: Mo, Ying et al. “C-ICL: Contrastive In-context Learning for Information Extraction.” *ArXiv* abs/2402.11254 (2024): n. pag.
[^27]: Qin, Chengwei et al. “In-Context Learning with Iterative Demonstration Selection.” *ArXiv* abs/2310.09881 (2023): n. pag.
[^28]:  Liu, Jiachang et al. “What Makes Good In-Context Examples for GPT-3?” *Workshop on Knowledge Extraction and Integration for Deep Learning Architectures; Deep Learning Inside Out* (2021).
[^29]: Zhang, Shaokun et al. “IDEAL: Influence-Driven Selective Annotations Empower In-Context Learners in Large Language Models.” *ArXiv* abs/2310.10873 (2023): n. pag.
[^30]: Mavromatis, Costas et al. “Which Examples to Annotate for In-Context Learning? Towards Effective and Efficient Selection.” *ArXiv* abs/2310.20046 (2023): n. pag.
[^31]: Wingate, David et al. “Prompt Compression and Contrastive Conditioning for Controllability and Toxicity Reduction in Language Models.” *Conference on Empirical Methods in Natural Language Processing* (2022).
[^32]: Chevalier, Alexis et al. “Adapting Language Models to Compress Contexts.” *ArXiv* abs/2305.14788 (2023): n. pag.
[^33]: Ge, Tao et al. “In-context Autoencoder for Context Compression in a Large Language Model.” *ArXiv* abs/2307.06945 (2023): n. pag.
[^34]: Mu, Jesse et al. “Learning to Compress Prompts with Gist Tokens.” *ArXiv* abs/2304.08467 (2023): n. pag.
[^35]: Gao, Jun. “Unifying Demonstration Selection and Compression for In-Context Learning.” *ArXiv* abs/2405.17062 (2024): n. pag.

