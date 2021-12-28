# Fast Non-parametric Multimodal Word Embeddings

We demonstrate a fast and simple DPGMM based method to find probabilistic word embeddings for any given word.
We avoid large training pipelines as used by [Athiwaratkun and Wilson](https://arxiv.org/abs/1704.08424) 
Instead, we find contextualized word embeddings for all contexts of each word and cluster them ising DPGMM.

We use the UKWAC dataset, which can be found [here](https://corpora.dipintra.it/wac/ukwac.html) for getting the embeddings.
And the Word-Entailment dataset, which can be found [here](https://marcobaroni.org/publications/bbds-eacl2012.pdf) for testing our 
Gaussians.

Our benchmark is using averaging context of word2vec. We also compare our results with the work by Athiwaratkun and Wilson, where
they used GMMs instead of DPGMM.
