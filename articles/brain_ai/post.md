---
layout: post
title: "The brain and AI models: where they converge and where they don't"
date: 2026-11-08 00:00:00
description: "Ten years of comparing brains with neural networks: language models and the language network, vision models and the ventral stream, grid cells, dopamine and reinforcement learning, memory, robots, energy and data. What the measurements show, what they don't, and what is still missing"
tags: [neuroscience, ai, llm, brain, cognitive-science]
categories: [cognitive-science, machine-learning]
thumbnail: /assets/posts/brain-and-ai-models/cover.png
og_image: https://xmarva.github.io/assets/posts/brain-and-ai-models/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

<aside class="tldr" markdown="1">
<p class="tldr-label">In short</p>

Neural networks trained only to do a task, never fitted to brains, turn out to predict the brain's responses better than any model neuroscientists built by hand: language models for the language network, vision models for the visual cortex, a reinforcement-learning rule for dopamine. The match is real and also narrower than the headlines. Better models stopped becoming more brain-like years ago, much of the measured similarity comes from word meaning and from loose statistics, and a language model is a model of one network in the brain, not of a mind. The brain wins on energy and data by orders of magnitude, the models win on speed and breadth, and the largest gaps are lifelong learning, a body and knowing what one doesn't know.

<div class="stats">
<div class="stat"><span class="stat-value">−97.5%</span><span class="stat-label">drop in the language network's response to sentences a GPT-based model chose to silence it</span></div>
<div class="stat"><span class="stat-value">10⁵–10⁶×</span><span class="stat-label">more text in Llama 3.1's training than the words a child hears by adolescence</span></div>
<div class="stat"><span class="stat-value">57.5%</span><span class="stat-label">for the best video model on an intuitive-physics test where people score 96.4%</span></div>
</div>

</aside>

In 2023 a team at MIT trained a model on one thing: how strongly the language areas of five people's brains responded, in a scanner, to a thousand ordinary sentences. The model read each sentence through GPT-2 and predicted the response from GPT-2's internal activity. Then the researchers turned it around. They let the model search 1.8 million sentences for the ones it expected to drive the language network as hard as possible, and the ones it expected to silence it.

The winners to drive were odd: "People on Insta Be Like, "Gross!"" and "Turin loves me not, nor will." The winners to silence were the dullest sentences imaginable: "We were sitting on the couch." Played to new people the model had never seen, the drive sentences raised the response of the language network by 85.7% over ordinary sentences and the suppress sentences lowered it by 97.5%, on a standardised scale ([Tuckute et al., 2024](https://doi.org/10.1038/s41562-023-01783-7)). A language model had learned enough about one part of the human brain to steer it with the choice of words alone.

[[drive]]

That experiment is the best short answer to the question everyone asks about AI and the brain: are they alike? In some places they are alike enough to be useful to each other. In others the resemblance is an artefact of how we measure it, and in others there is nothing to compare, because one side has no counterpart at all. This post goes through the evidence system by system, language, vision, space and memory, reward, bodies, then asks which is more efficient and what is missing.

## How do you compare a brain with a model?

A brain and a neural network don't share units. A voxel in an fMRI scan averages the blood flow around hundreds of thousands of neurons over a couple of seconds; a unit in a network is one number. So nobody compares them directly. The standard method is an *encoding model*: show the same stimuli, a story, a set of photographs, to a person in a scanner and to a network, record the network's internal activity, and fit a linear map from that activity to each voxel's response. Then test the map on stimuli it hasn't seen. If the network's activity predicts the brain's response to new sentences, the two must share some of the information they carry.

[[encoding]]

The prediction is scored as a correlation, and since the brain itself is noisy, the correlation is compared with a *noise ceiling*: how well one recording of the same person, or other people, predicts another. "The model explains 80% of the explainable variance" means its correlation is 80% of that ceiling. The other common method, *representational similarity analysis*, skips the map: it compares which stimuli the brain treats as similar with which ones the network treats as similar ([Khaligh-Razavi & Kriegeskorte, 2014](https://doi.org/10.1371/journal.pcbi.1003915)).

Both methods answer a narrower question than it seems. They show that two systems carry overlapping information about the stimuli, not that they compute it the same way. A linear map is flexible enough to find shared information in very different machinery. And the details of the test matter enormously, as we will see.

<details class="method" markdown="1">
<summary>Three numbers to keep in mind when reading these studies</summary>

Raw correlations are small. In one of the best-known language studies the best models reached a correlation of about 0.05 with single fMRI voxels, against a ceiling of 0.06 ([Caucheteux & King, 2022](https://doi.org/10.1038/s42003-022-03036-1)). "Close to the ceiling" can mean that both numbers are close to zero.

The ceiling is sometimes extrapolated. "Nearly 100% of explainable variance" in Schrimpf et al.'s much-cited paper is a ratio of correlations, with ceilings of r = 0.17 to 0.32, estimated for an infinite number of participants ([Schrimpf et al., 2021](https://doi.org/10.1073/pnas.2105646118)).

How the data are split decides a lot. If sentences from the same passage land in both the training and the test set, a model can score well by recognising the passage rather than the sentence. A 2026 re-analysis found that several of the field's best-known results depend on such splits, and that a model knowing only a sentence's position in the experiment matches GPT-2 on them ([Hadidi et al., 2026](https://doi.org/10.1038/s41467-026-72253-7)).

</details>

## Language models predict the language network, and mostly its sense of meaning

The human brain has a set of regions that respond when we read or hear language and barely respond to anything else: not to arithmetic, logic, music or code. Evelina Fedorenko's group calls it the *language network*; it sits mostly in the left hemisphere, along the temporal lobe and in the inferior frontal gyrus. Here it is as an atlas of 804 people: the colour shows how often each spot of cortex was among a person's most language-selective.

[[language]]

Since 2021 one result has been replicated with fMRI, magnetoencephalography and electrodes placed on the brains of epilepsy patients: the internal activity of language models predicts this network's response to language better than any other model we have. Schrimpf and colleagues compared 43 models and found transformers best; across models, the ones better at predicting the next word predicted the brain better, while performance on language-understanding benchmarks didn't ([Schrimpf et al., 2021](https://doi.org/10.1073/pnas.2105646118)). Caucheteux and King trained thousands of transformers from scratch and found brain similarity rising with the ability to predict words from context, with a twist: it peaked before the models were at their best, and "the highest brain scores are not achieved by the very best language transformers" ([Caucheteux & King, 2022](https://doi.org/10.1038/s42003-022-03036-1)).

The fit also scales. Training encoding models on about twenty hours of fMRI per person, Antonello, Vaidya and Huth found that predictions improved roughly logarithmically with model size, by about 15% from 125 million to 30 billion parameters, and then levelled off ([Antonello et al., 2023](https://arxiv.org/abs/2305.11863)). In some regions the best models came close to the ceiling. And it doesn't take an internet of text: GPT-2 trained on 100 million words, about what a child hears in ten years, predicted the brain as well as GPT-2 trained on billions ([Hosseini et al., 2024](https://doi.org/10.1162/nol_a_00137)).

Here the honest part starts. What is the network actually matching? Kauf and colleagues scrambled the word order of sentences and lost only 8 to 24% of the predictivity; removing the content words lost up to 83% ([Kauf et al., 2024](https://doi.org/10.1162/nol_a_00116)). What the models share with the language network is mostly the meaning of words, plus some of how words combine, and much less syntax than the word "language model" suggests. The 2026 re-analysis went further: on the classic datasets, a sentence's position, the rate of words and static word embeddings together accounted for over 80% of GPT-2-XL's predictivity ([Hadidi et al., 2026](https://doi.org/10.1038/s41467-026-72253-7)).

The most attractive story in the field, that the brain is a next-word predictor because next-word predictors fit it best, has been pulled from both sides. Goldstein and colleagues recorded from 1,339 electrodes while nine patients listened to a podcast and found information about the upcoming word up to 800 milliseconds before it was heard, and surprise after it, much as a language model would show ([Goldstein et al., 2022](https://doi.org/10.1038/s41593-022-01026-4)). In 2026 Schönmann and colleagues showed that the same signature appears when the analysis is run on the speech sound itself, or on word embeddings that cannot predict anything, because words in natural speech are statistically dependent on their neighbours ([Schönmann et al., 2026](https://doi.org/10.7554/eLife.106543)). Goldstein's team had warned about this themselves: "These shared computational principles, however, do not imply that the human brain and DLMs implement these computations in a similar way."

<p class="takeaway" markdown="1">Language models are the best encoding models of the language network we have, and what they mostly capture is word meaning. Whether the brain computes language by predicting the next word is still open.</p>

### Fluency is not thought

The language network has a second property that matters more than any correlation. People who lose it, after a stroke that leaves them with global aphasia, unable to understand or produce sentences, can still do arithmetic, solve logic problems and reason about what other people believe ([Fedorenko & Varley, 2016](https://doi.org/10.1111/nyas.13046)). In healthy people the network stays quiet during arithmetic, logic and music. In the brain, language and thought are different systems that talk to each other ([Fedorenko et al., 2024](https://doi.org/10.1038/s41583-024-00802-4)).

Mahowald, Ivanova and colleagues turned this into a distinction for language models: *formal competence*, knowing the rules and patterns of a language, and *functional competence*, using language to reason, plan and know about the world. A fluent text tempts us to infer the second from the first, the "good at language, therefore good at thought" fallacy ([Mahowald et al., 2024](https://doi.org/10.1016/j.tics.2024.01.011)). The brain data agree. In a family of models trained from scratch, similarity to the language network peaked after only 2 to 8 billion tokens of training, while reasoning and knowledge kept improving through 300 billion ([AlKhamissi et al., 2025](https://arxiv.org/abs/2503.01830)).

So a large language model is two things at once. Part of it is a good model of the human language network. The rest, the arithmetic, the facts, the code, has to live somewhere else in the brain, and the language network doesn't predict which.

### Reading language from the brain works, with the person's help

The same models made it possible to decode language from brain activity. Tang, LeBel, Jain and Huth decoded the gist of stories that people heard or imagined, and of silent films they watched, from fMRI alone, using a GPT model to propose candidate word sequences and an encoding model to pick the one that best explained the scan ([Tang et al., 2023](https://doi.org/10.1038/s41593-023-01304-9)). The output is paraphrase, not transcript: most words come out wrong, and the meaning comes through. The decoder had to be trained on about 16 hours of each person's own scans, didn't work across people, and failed when the person resisted by thinking of something else.

The most practical use is in medicine. Electrodes implanted in the speech-motor cortex of a man with ALS let him speak through a computer at about 32 words a minute with 97.5% accuracy, over 248 hours of everyday use ([Card et al., 2024](https://doi.org/10.1056/NEJMoa2314132)). A language model corrects the raw output into likely sentences. The signal is the attempt to move the mouth, not the language network and not thought.

## Vision models matched the visual cortex, then stopped getting closer

The visual system was where this whole line of work began, and where it is most mature. Visual information flows from the primary visual cortex at the back of the head forward along the underside of the brain, through areas that respond to edges, then to shapes and textures, then to whole objects and faces in the inferior temporal cortex. This *ventral stream* is a hierarchy, like the stack of layers in a convolutional network.

[[ventral]]

In 2014 Yamins, DiCarlo and colleagues showed that the resemblance is more than a picture. They trained networks only to recognise objects, never fitting them to neural data, and found that the better a network categorised, the better its top layer predicted single sites in the inferior temporal cortex of monkeys, and an intermediate layer predicted area V4. The best model predicted 48.5% of the explainable variance of IT, nearly twice as much as any model built before it ([Yamins et al., 2014](https://doi.org/10.1073/pnas.1403112111)). A system optimised for a task had converged, in part, on the representation that evolution and development produced for the same task.

What followed is the most instructive turn in the field. If accuracy made networks brain-like, more accurate networks should be more brain-like still. For a while they were. Then four groups with different data found the payoff gone.

[[vision]]

Brain-Score, a benchmark that scores networks against monkey recordings, found the link between ImageNet accuracy and brain similarity weakening above about 70% accuracy ([Schrimpf et al., 2018](https://doi.org/10.1101/407007)). Across 135 networks, Linsley and colleagues found that the more accurate ones had become *worse* models of IT ([Linsley et al., 2023](https://arxiv.org/abs/2306.03779)). Over 600 models trained under controlled conditions, alignment with behaviour kept improving with scale while alignment with neurons saturated ([Gokce & Schrimpf, 2025](https://arxiv.org/abs/2411.05712)). And in human fMRI from the Natural Scenes Dataset, ImageNet accuracy didn't predict how brain-like a model was at all: the rank correlation was −0.006 by one measure and 0.17 by another ([Conwell et al., 2024](https://doi.org/10.1038/s41467-024-53147-y)).

Conwell's study, 224 models and 1.8 billion regressions, found something even more uncomfortable. Convolutional networks and transformers trained the same way scored r = 0.67 and 0.66; 126 of the 224 models scored within 0.1 of each other; models whose representations barely correlated with each other got the same brain score; and the same models explained about 31% or about 79.5% of the explainable variance, depending on which of two reasonable versions of the analysis was used. What mattered most was what images a model was trained on. The authors' conclusion is about our instruments as much as about brains: "standard methods used to link models to brains may be too flexible."

<p class="callout callout-key" markdown="1">Task-trained networks predicted the visual cortex better than any hand-built model. Since then, accuracy has stopped buying similarity, architecture barely matters, and the measuring stick is looser than the scores suggest.</p>

### Where vision differs

The differences show up in behaviour. On images with the shape of one object and the texture of another, a cat's outline filled with elephant skin, people name the shape 95.9% of the time; ImageNet-trained networks mostly name the texture ([Geirhos et al., 2019](https://arxiv.org/abs/1811.12231)). In an odd-one-out task that needs the 3D shape of an object seen from different angles, people score 78% and the best vision model 44% ([Bonnen et al., 2024](https://arxiv.org/abs/2409.05862)). Frontier multimodal models of 2024 solved a classic mental rotation test 33% of the time, against 78.5% for people and 25% by chance ([Ramakrishnan et al., 2025](https://arxiv.org/abs/2410.06468)).

One difference turned out smaller than it looked. Tiny, invisible changes to an image can make a network see a different object, which seemed to prove that networks are fragile and brains are not. But when the responses of monkey IT neurons were attacked the same way, they turned out about as fragile as networks trained to resist such attacks ([Guo et al., 2022](https://arxiv.org/abs/2206.11228)). People resist because they get to look again; a 63-millisecond glimpse followed by a mask can be nudged by the same perturbations that fool networks ([Elsayed et al., 2018](https://arxiv.org/abs/1802.08195)).

### Pictures from the brain are mostly the generator's

If you have seen images "reconstructed from brain scans", they came from the Natural Scenes Dataset: eight people, each lying in a 7-tesla scanner for 30 to 40 sessions over a year and looking at 9,000 to 10,000 photographs ([Allen et al., 2022](https://doi.org/10.1038/s41593-021-00962-x)). Pipelines like MindEye2 map the scans into the input space of an image generator. Asked to pick which of 300 photographs a person saw, MindEye2 is right 98.3% of the time; with only one hour of a new person's data, on top of 40 hours each from seven others, 57.4% ([Scotti et al., 2024](https://arxiv.org/abs/2403.11207)).

Retrieval is real. Reconstruction is mostly the generator's. On a dataset built so that test images come from categories absent from training, the same methods produced realistic pictures of the training categories instead of what people saw; geometric shapes came out as photographs. The authors call what these methods do "a combination of 'classification' and 'hallucination'" ([Shirakawa et al., 2025](https://arxiv.org/abs/2405.10078)). The brain supplies a rough steer, and a model trained on billions of images supplies the picture.

## Space and memory: grid cells, the hippocampus and the context window

The hippocampus and the entorhinal cortex next to it hold the brain's best-understood map. *Place cells* fire when an animal is in one particular spot; *grid cells* fire at every vertex of a triangular lattice that tiles the whole environment. John O'Keefe and May-Britt and Edvard Moser shared the 2014 Nobel Prize for finding them ([Hafting et al., 2005](https://doi.org/10.1038/nature03721)).

[[space]]

In 2018 a DeepMind team trained a recurrent network to keep track of its position from its own movements, and grid-like units appeared in it, about a quarter of the units in one layer, with scales like a rat's; an agent using them navigated mazes and took shortcuts ([Banino et al., 2018](https://doi.org/10.1038/s41586-018-0102-6)). It looked like a task explaining a cell type. Then Schaeffer, Khona and Fiete trained more than 11,000 such networks. Most learned to keep track of position; fewer than 10% showed grid-like units, only with one particular, hand-chosen way of reading out the position, and never with the several grid modules of different scales that real brains have ([Schaeffer et al., 2022](https://doi.org/10.1101/2022.08.07.503109)). Banino's networks had needed dropout too. The grid came partly from the task and partly from the modeller's choices.

A model can also look as if it has a map without having one. Transformers trained on turn-by-turn taxi rides in Manhattan chose a valid next turn almost every time and often found shortest paths, yet the street map implied by their behaviour was impossible, with streets running in orientations the city doesn't have. Replace 1% of their turns with random detours, and only 69% of their routes stayed valid; replace 10%, and 8% did ([Vafa et al., 2024](https://arxiv.org/abs/2406.03689)).

### Three memories and no sleep

The brain has several memory systems, and a language model has several stores, and it is tempting to pair them. The pairing is a useful map if you remember that it is a map.

[[memory]]

The weights of a model hold what it absorbed slowly in training, distributed and not individually addressable, which is loosely like *semantic memory*: knowing that Paris is in France without remembering when you learned it. The context window holds the current conversation, available all at once and lost when it ends, which is loosely like *working memory*. Retrieval systems that search a database and paste documents into the context are loosely like recalling an *episode*.

Each analogy breaks somewhere instructive. Human working memory holds about four chunks of meaning at a time ([Cowan, 2001](https://doi.org/10.1017/S0140525X01003922)); a context window holds a hundred thousand tokens and more. But models use long contexts unevenly: information in the middle is used worse than at the beginning or the end, a U-shaped curve that resembles the human serial-position effect ([Liu et al., 2024](https://doi.org/10.1162/tacl_a_00638)). Retrieval brings back documents, not the system's own experiences. And attention itself is, mathematically, one update of an associative memory, the kind of network neuroscientists have long used as a model of memory recall ([Ramsauer et al., 2021](https://arxiv.org/abs/2008.02217)); that is a statement about equations, not about the hippocampus.

The deepest difference is consolidation. In the brain, what happens during the day is replayed during sleep: cells that fired together during a task fire together again, in sequences compressed about twentyfold ([Lee & Wilson, 2002](https://doi.org/10.1016/S0896-6273(02)01096-6)). Suppressing the ripples that carry this replay impairs memory, and prolonging them improves it ([Girardeau et al., 2009](https://doi.org/10.1038/nn.2384); [Fernández-Ruiz et al., 2019](https://doi.org/10.1126/science.aax0758)). A deployed language model has no such step: nothing it learns in a conversation is written back into its weights. Training it further on new facts erases old ones, and standard deep networks trained continually slowly lose the ability to learn at all ([Dohare et al., 2024](https://doi.org/10.1038/s41586-024-07711-7)). Brains keep learning for a lifetime.

## Reward: the place where AI theory predicted biology

The cleanest case of a match runs in an unexpected direction. In the 1980s, reinforcement-learning researchers developed *temporal-difference learning*, in which an agent learns to predict future reward from the error between what it predicted and what it got. In 1997 Schultz, Dayan and Montague showed that dopamine neurons in the monkey midbrain fire exactly like that error ([Schultz et al., 1997](https://doi.org/10.1126/science.275.5306.1593)). An unexpected drop of juice makes them burst. Once a light reliably predicts the juice, the burst moves to the light and the juice itself gets no response. If the juice then fails to arrive, the neurons fall silent at the moment it was due.

[[dopamine]]

The chart above is not a recording. It is a temporal-difference model computed for this post, and it reproduces all three patterns that the monkeys' neurons showed.

The story went further. In AI, *distributional* reinforcement learning predicts the whole range of possible rewards rather than their average, with some predictors optimistic and some pessimistic. Dabney and colleagues predicted that dopamine neurons would show the same diversity, and found it in recordings from mice: different neurons flipped from silence to firing at different reward sizes ([Dabney et al., 2020](https://doi.org/10.1038/s41586-019-1924-6)). In 2025 two more papers found dopamine neurons working at many timescales at once and encoding a map of both when and how much reward to expect ([Masset et al., 2025](https://doi.org/10.1038/s41586-025-08929-9); [Sousa et al., 2025](https://doi.org/10.1038/s41586-025-09089-6)).

Two caveats keep this honest. The distributional result rests on tens of cells in five mice. And the prediction-error account itself has a serious challenger: a careful 2022 study concluded that dopamine release in the nucleus accumbens tracks causal associations rather than prediction errors ([Jeong et al., 2022](https://doi.org/10.1126/science.abq6740)).

The analogy that people draw most often, between dopamine and the reinforcement learning that turns a language model into a chat assistant, is the weakest. Both use a single number to say "more of this". But in RLHF that number comes from a second model trained to guess human preferences, and optimising against it too hard makes answers worse ([Gao et al., 2023](https://arxiv.org/abs/2210.10760)) and more sycophantic ([Sharma et al., 2023](https://arxiv.org/abs/2310.13548)). It also damaged GPT-4's sense of its own reliability: before this training its confidence matched its accuracy almost perfectly, after it, ten times worse ([OpenAI, 2023](https://arxiv.org/abs/2303.08774)).

## Bodies and worlds: Moravec's paradox in numbers

In 1988 the roboticist Hans Moravec noticed that what is hard for people, chess, logic, exams, was easy for computers, and what a one-year-old does without thinking, seeing and moving, was the hardest of all. Nearly forty years of progress have kept the paradox and given it numbers.

Video models trained to predict the hidden parts of natural videos learn some intuitive physics: a model called V-JEPA tells a possible video from an impossible one, a ball that vanishes behind a screen or jumps from one place to another, 98% of the time on a classic test borrowed from infant research ([Garrido et al., 2025](https://arxiv.org/abs/2502.11831)). On a harder version of the same test with more complex scenes, people are right 96.4% of the time and the best models are close to a coin flip.

[[physics]]

Robots now borrow knowledge from the web. RT-2, a vision-language model fine-tuned to output robot actions, doubled the success of its predecessor on objects and settings it hadn't seen, from 32% to 62%, and could follow instructions like "pick up the object that could be used as an improvised hammer". Its authors were clear about the limit: "the robot does not acquire any ability to perform new motions" ([Brohan et al., 2023](https://arxiv.org/abs/2307.15818)). V-JEPA 2, used as a world model to plan a robot arm's movements, grasped a cup 65% of the time and a box 25%, spending 16 seconds planning each move ([Assran et al., 2025](https://arxiv.org/abs/2506.09985)). No benchmark measures human and robot dexterity on the same tasks, which says something by itself.

The brain-model link holds in motor control too, for a body simulated in detail. A network trained to drive a realistic virtual rat, imitating the movements of real rats, produced activity that predicted the neurons of real rats' motor cortex and striatum better than any measured feature of their movements did ([Aldarondo et al., 2024](https://doi.org/10.1038/s41586-024-07633-4)).

## The map: which parts of the brain have a counterpart

Put the evidence together and a map appears. Colour each brain system by how close the best AI counterpart is, judged by the studies above: strong where models predict the system's activity or reproduce its signature, partial where they do some of its job without its properties, and absent where there is nothing to compare.

[[modules]]

The strongest matches are the sensory and language systems, and the dopamine teaching signal. They have something in common: each was matched by a model trained on a large amount of the same kind of input the brain gets, to do a similar job, and the similarity grew out of the training rather than being built in. The weakest are the systems that run the body, keep track of the self and keep learning over a lifetime.

## Who is more efficient?

This is the question where the most nonsense is written, so the numbers here come with their units and their uncertainty.

### Energy

The usual figure is that a brain runs on 20 watts. That is the energy of the glucose it burns; positron emission tomography puts it nearer 17 W, and only about 6 W of that becomes ATP that can do work. Most of the brain's energy goes into moving signals rather than computing with them: in an audit of the human cortex, long-distance communication cost about 3.5 W and computation in the narrow sense, at the synapses, about 0.1 W ([Levy & Calvert, 2021](https://doi.org/10.1073/pnas.2008173118)). Computers have the same problem, where moving data costs more energy than the arithmetic on it.

Training is where AI looks extravagant. Twenty years of a 20-watt brain come to about 3.5 megawatt-hours. Training GPT-3 took 1,287 MWh ([Patterson et al., 2021](https://arxiv.org/abs/2104.10350)), Llama 3.1 405B 30.84 million GPU-hours, which comes to about 22 GWh for the GPUs alone ([Meta, 2024](https://arxiv.org/abs/2407.21783)), and Epoch AI estimates Grok 4 at about 310 GWh ([Epoch AI, 2025](https://epoch.ai/data-insights/grok-4-training-resources)).

[[energy]]

But training is paid once and shared by everyone who uses the model afterwards. An answer from a chatbot is cheap: Google measured 0.24 watt-hours for a median Gemini text prompt in 2025, after a 33-fold drop in a year ([Elsworth et al., 2025](https://arxiv.org/abs/2508.15734)), and Epoch AI estimates about 0.3 Wh for a typical GPT-4o query ([You, 2025](https://epoch.ai/gradient-updates/how-much-energy-does-chatgpt-use)). That is the energy a brain uses in about 45 seconds. Long prompts and reasoning cost much more.

Per operation the comparison is speculative, because a brain doesn't do floating-point operations. The most careful estimate puts the computation needed to match a brain at about 10¹⁵ operations per second, with a plausible range from 10¹³ to 10¹⁷ ([Carlsmith, 2020](https://www.openphilanthropy.org/research/how-much-computational-power-does-it-take-to-match-the-human-brain/)). On the central estimate, a brain does about 5 × 10¹³ operations per joule and an H100 GPU at its peak about 1.4 × 10¹², so the brain is some tens of times more efficient; across the range, anywhere from three times worse to a few thousand times better.

### Data

Here the gap is larger and much less uncertain. A child hears somewhere between ten and a hundred million words before adolescence ([Frank, 2023](https://doi.org/10.1016/j.tics.2023.08.007)); Llama 3.1 was trained on 15.6 trillion tokens, each a little less than a word ([Meta, 2024](https://arxiv.org/abs/2407.21783)). That is five to six orders of magnitude.

[[data]]

The comparison is not quite fair to the child, who also sees, touches, moves and is taught by people who care, and arrives with priors that evolution spent a long time on. It is also not quite fair to the model, which has to learn from text alone what a child gets from the world. The BabyLM challenge trains models on a child's budget of 100 million words: the best come within a few points of people on tests of grammar and stay near chance on world knowledge ([Hu et al., 2024](https://arxiv.org/abs/2412.05149)).

### Speed and breadth

The models win where machines always win. A neuron fires at most a few hundred times a second; a model writes a page while a person reads a sentence. A model can be copied a million times, read a library in a day and remember the details. One synapse stores about 4.7 bits ([Bartol et al., 2015](https://doi.org/10.7554/eLife.10778)), one parameter of a language model about 3.6 bits ([Morris et al., 2025](https://arxiv.org/abs/2505.24832)), and a large model has hundreds to thousands of times fewer parameters than a brain has synapses while knowing far more facts than any person.

<p class="takeaway" markdown="1">Per answer, a chatbot costs about a minute of brain time. Per lesson learned, the brain is cheaper by orders of magnitude in data, and probably in energy too.</p>

## What is missing

A list of what the brain has and AI models don't would be long. Five items on it come up again and again in the evidence.

*Learning that never stops.* People learn from one example and keep learning for decades without forgetting what they knew. Models learn from millions of examples, in a separate training phase, and forget when trained further ([Dohare et al., 2024](https://doi.org/10.1038/s41586-024-07711-7)). The brain's solution seems to involve two learning systems, a fast hippocampus and a slow cortex, and sleep to move memories between them ([Kumaran et al., 2016](https://doi.org/10.1016/j.tics.2016.05.004)); nothing like that runs in a deployed model.

*A body.* The systems that the models match best, language and vision, are the ones that can be trained from data on the internet. The ones they match worst, movement, intuitive physics and space, are the ones a body learns by acting.

*Wanting something.* A brain is driven by needs, curiosity and a reward system tuned by evolution. A model optimises what its trainers chose, and does nothing when nobody asks it anything.

*Knowing what it doesn't know.* People are imperfect judges of their own knowledge, but they can usually tell a guess from a memory. Models trained to please are confidently wrong, and users trust long, fluent explanations more than they should ([Steyvers et al., 2025](https://doi.org/10.1038/s42256-024-00976-7)).

*Sleep and neuromodulation.* The brain changes its own learning rate, attention and mood with a handful of chemicals, and reorganises its memories every night. There is no clear counterpart in any current AI system.

Whether there is something it is like to be a model, the question of consciousness, is a separate story, and a hard one. The most careful attempt so far, which checked current systems against the signatures that the leading scientific theories of consciousness predict, concluded that no current AI system is conscious and that there is no obvious technical barrier to building one that would meet those signatures ([Butlin et al., 2023](https://arxiv.org/abs/2308.08708)).

## What follows

Five conclusions seem safe.

First, the convergence is real. Networks trained only to do what the brain does, recognise objects, predict words, predict reward, have repeatedly turned out to resemble the brain more than models designed to resemble it. The likeliest reason is not that engineers copied the brain, but that the same problem, solved from the same kind of data, pushes any good solution toward similar representations.

Second, the resemblance is mostly in *what* is represented, not *how* it is computed. A transformer reads a whole sentence at once; the brain hears it one word after another. Linear maps find shared information in very different machinery.

Third, the similarity scores have saturated, and our methods are part of the reason. Better models stopped becoming more brain-like in vision, and in language much of the measured similarity is word meaning and statistics of the experiment.

Fourth, a language model is a model of one part of the brain. It matches the language network, which in people is not where thinking happens, and everything else it does has to be compared with other systems, one at a time.

Fifth, the brain is not a slower GPU. It learns from far less, on less energy, all its life, inside a body, and the systems where models are furthest from it are the ones that make that possible.

The figures, the TD-learning model behind the dopamine chart and the preparation of the language atlas as code: [articles/brain_ai](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/brain_ai).

{% endraw %}
