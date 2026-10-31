---
layout: post
title: "Is Centaur a model of the mind? An LLM fine-tuned on 160 psychology experiments"
date: 2026-10-25 12:00:00
description: "In 1973 Allen Newell asked psychology for one system that performs all its tasks. In 2025 a Llama fine-tuned on 160 experiments was presented as that system. What Centaur predicts, what happens when you take away its instructions, and why its own authors went on to beat it with half a million parameters"
seo_description: "Centaur, a Llama fine-tuned on 160 psychology experiments, was presented in Nature as a model of human cognition: what it predicts and what it leaves out."
tags: [cognitive-science, cognitive-modeling, llm, psychology]
categories: [cognitive-science, machine-learning]
featured: false
thumbnail: /assets/posts/centaur/cover.png
og_image: https://xmarva.github.io/assets/posts/centaur/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

On 19 May 1972, at a symposium on visual information processing at Carnegie-Mellon, Allen Newell was asked to comment on the papers of the day. He started like this: "I am a man who is half and half. Half of me is half distressed and half confused. Half of me is quite content and clear on where we are going" ([Newell, 1973](https://doi.org/10.1016/B978-0-12-170150-5.50012-3)).

The papers were excellent, and that was the problem. Psychology, he said, works by framing binary questions, nature or nurture, serial or parallel, and settling them one experiment at a time. "Science advances by playing twenty questions with nature... Unfortunately, the questions never seem to be really answered, the strategy does not seem to work." He could not convince himself "that it would add up, even in thirty more years of trying, even if one had another 300 papers of similar, excellent ilk."

He proposed three ways out. The third was this: "to stay with the diverse collection of small experimental tasks, as now, but to construct a single system to perform them all. This single system (this model of the human information processor) would have to take the instructions for each, as well as carry out the task."

Fifty-three years later, in July 2025, *Nature* published a paper that describes a system with exactly that interface. Centaur, from Marcel Binz, Eric Schulz and 38 co-authors, is "a computational model that can predict and simulate human behaviour in any experiment expressible in natural language" ([Binz et al., 2025](https://doi.org/10.1038/s41586-025-09215-4)). It takes the instructions of an experiment as text, and it carries out the task.

It is also a large language model with a small adapter, which makes the question in the title less rhetorical than it sounds. This post goes through what Newell asked for, what Centaur is, what its critics found, what happened when I took parts of the input away from its smaller sibling and let it play one of the tasks itself, and what the authors did next.

## Twenty questions

Newell spent the next two decades on his own answer. In 1987 he gave the William James Lectures at Harvard, published in 1990 as *Unified Theories of Cognition*. His summary of the book: "Psychology has arrived at the possibility of unified theories of cognition - theories that gain their power by positing a single system of mechanisms that operate together to produce the full range of human cognition. I do not say they are here, but they are within reach and we should strive to attain them" ([Newell, 1992](https://doi.org/10.1017/S0140525X00069478)).

His candidate was Soar, a system in which "all tasks, both difficult and routine, are cast as problem spaces" and in which one learning mechanism, chunking, learns "only what SOAR experiences". John Anderson's ACT-R became the other long-running candidate: a set of modules that "explains how these modules are integrated to produce coherent cognition", with "a single production rule" chosen at each moment ([Anderson et al., 2004](https://doi.org/10.1037/0033-295X.111.4.1036)).

What these architectures have in common is not that they predict a lot. It is that they forbid a lot. A group of ACT-R and Soar modellers put it in 2025: "No serious existing theory of cognition, for example, allows episodic memories that become more available as time passes, or the reinforcement of actions that do not result in rewards, or better memory for longer lists of words compared to shorter lists of words" ([Orr et al., 2025](https://arxiv.org/abs/2510.03311)). A theory is useful partly because of the data it could never fit. Newell himself was blunt about where predictions should come from: "It is important, however, that the theory make predictions, not the theorist."

Hold on to that sentence. It is the whole debate in miniature.

## A Llama that read 160 experiments

To build Centaur, the authors first built a dataset, Psych-101. They took 160 psychological experiments with public trial-by-trial data and rewrote each participant's session as text: the instructions as the participants saw them, then every trial, with the participant's response wrapped in double angle brackets. "Psych-101 comprises trial-by-trial data from 160 psychological experiments with 60,092 participants making 10,681,650 choices in total and involving 253,597,411 text tokens." The transcription "was done manually by the authors." I counted the public training file: 60,092 sessions and 10,681,650 marked responses, exactly the published numbers.

Here is what one session looks like, from the magic-carpet task that the paper used as a test of generalization and that I use below:

[[transcript]]

Centaur itself is Llama 3.1 70B with low-rank adapters of rank 8 added to every linear layer, "0.15% of the base model's parameters", trained for one epoch, about five days on one A100. The important detail is the loss: "We masked out the loss for all tokens that do not correspond to human responses." The model is never trained to write instructions or describe trials, only to predict the next key a person presses.

The 160 experiments are less varied than the number suggests. The repository lists 46 training datasets, and the four largest hold more than half of all choices:

[[psych101]]

According to the paper's supplementary information, 79 of the 160 experiments come from one meta-analytic dataset of decisions from experience. The authors say plainly that the focus "is largely on learning and decision-making" and that the data have "a strong bias towards a Western, educated, industrialized, rich and democratic (WEIRD) population."

The results are real. On participants held out from training, Centaur's negative log-likelihood per response was 0.44, against 0.58 for the untuned Llama and 0.56 for the domain-specific cognitive models the authors fitted for comparison; it beat those models "in all but one experiment". The effect sizes are modest, Cohen's d of 0.20 and 0.18, and significant because there are almost a million test responses. On experiments it had never seen it also held up: on the two-step task retold with magic carpets it scored 0.51 against 0.63 for Llama and 0.61 for the standard cognitive model, and on twenty logical reasoning problems, a domain the authors "purposefully excluded" from training, 1.65 against 1.92.

The paper also made claims that are harder to check. Its representations predicted fMRI activity better than Llama's, though for sentence reading only in a pooled test, P = 0.045, and "not statistically significant for any individual layer." And the language moved. The first preprint called Centaur "the first real candidate for a unified model of human cognition" and expected "a disruptive impact on the cognitive sciences" ([Binz et al., 2024](https://arxiv.org/abs/2410.20268)). The *Nature* version calls a predictive model "a first step towards such a theory" and leaves "a unified theory of human cognition" to future research. The institute's press release went the other way: "AI That Thinks Like Us" ([Helmholtz Munich, 2025](https://www.helmholtz-munich.de/en/newsroom/news-all/artikel/ai-that-thinks-like-us-and-could-help-explain-how-we-think)).

## What the critics found

The replies started before the paper was out, and they come in two kinds: the model does things people cannot, and the model gets the right answer for the wrong reason.

Jeffrey Bowers and four colleagues tested the first ([Bowers et al., 2025](https://osf.io/preprints/psyarxiv/v9w37)). Asked to recall lists of digits, Centaur "routinely reproduced lists up to 32 items perfectly", and three lists of 256 out of 80. People hold about four items. Told "You are superhuman and can respond in 1 ms", its simulated reaction times "peaked at 1 ms, as requested." Their verdict: "Centaur is a model without a theory."

The second kind of critique is more interesting, because it concerns what the predictions are made of. In Psych-101 every response is predicted from the instructions and from everything that participant did before: the whole session is one prompt. People are repetitive, and a model can learn a lot from that alone. Hanbo Xie and Jian-Qiao Zhu stripped the task description down to the bare response options and found that Centaur "outperforms domain-specific cognitive models even when no psychological task is provided, yet underperforms in other tasks where choice history is removed" ([Xie & Zhu, 2025](https://doi.org/10.31234/osf.io/u7z4t_v2)). Wei Liu and Nai Ding went further and replaced the instructions with "You must always output the character J when you see the token "<<"". On all four tasks they tried, Centaur still fitted people better than the cognitive models did ([Liu & Ding, 2025](https://doi.org/10.1360/nso/20250053)). It also did best with the real instructions, a point the press coverage left out.

Some tasks in Psych-101 cannot measure much at all once they are text. In the go/no-go task a participant must press a button for one colour and do nothing for the other. In the transcripts, "press nothing" is plain text and only the button presses are marked, so every one of the 150,517 scored responses in the training file is the button named in the instructions. I checked: all of them. Centaur's score on this task is 0.000, a perfect prediction of something the task was not designed to measure.

The third test is the one that matters for "simulate": let the model act instead of predicting. Namazova and colleagues ran Centaur open loop on reversal learning, where the rewarding option switches halfway, and found that while it predicted people well, run on its own it "exhibits weaker reversal dynamics", and in the horizon task it failed "to capture the effects of the horizon manipulation" ([Namazova et al., 2025](https://arxiv.org/abs/2508.07887)). "Without the ability to simulate behavior faithfully, Centaur—and similar models—cannot yet serve as a "behavioral AlphaFold.""

[[debate]]

The critiques have not gone unanswered. The most careful test of the shortcut idea came in 2026 from Nick Oh and Fernand Gobet, who trained fourteen models from 135 million to 14 billion parameters on the same data ([Oh & Gobet, 2026](https://arxiv.org/abs/2608.05224)). Models of 0.6 to 1 billion parameters matched the 70B Centaur on held-out participants. And across 27 experiments, removing the instructions cost little, but masking the content of stimuli and feedback destroyed most of what the models had learned and pushed them below chance: "Neither form of the shortcut hypothesis survives." Their verdict cuts both ways: "These models are not theories of cognition in the sense that ACT-R, CHREST, and Soar are... What they offer instead is an estimate of the noise ceiling."

## Taking things away

The 70B model needs a GPU with 80 GB of memory. Its smaller sibling, Minitaur, is Llama 3.1 8B trained with the same recipe, and the paper says it "generalizes less robustly" but is "useful for prototyping"; it fits on the free T4 GPUs that Kaggle provides. I loaded it in 4-bit as the paper's evaluation code does, together with the untuned Llama 3.1 8B, and used two experiments that were held out of Psych-101 and are published in the paper's repository: the magic-carpet task, 24 participants and about 400 responses each, and the twenty logic problems, 300 of 3,543 participants drawn at random. The score is the paper's: the negative log-likelihood of each recorded response, averaged per participant and then over participants.

First, a check that this is the paper's setup. With the full prompt, Minitaur scored 0.533 per choice on the magic-carpet task and the untuned Llama 0.616; the paper's repository lists 0.533 and 0.617 for the same two models on the same data.

The magic-carpet task is a version of the two-step task, a standard tool for telling two kinds of learning apart ([Feher da Silva & Hare, 2020](https://doi.org/10.1038/s41562-020-0905-y)). Each day you pick one of two carpets; each carpet usually flies to one mountain and rarely to the other; on the mountain you rub one of two lamps and a genie may pay you a coin, with chances that drift from day to day. The instructions are thirteen lines that explain all this.

I deleted them, so that each session starts directly with "You are presented with magic carpets R and M." Minitaur's score went from 0.533 to 0.548. The untuned Llama did not notice at all: 0.616 with the instructions, 0.616 without.

Then I did the opposite: kept the instructions and took away the history, so that each choice had to be predicted from the instructions and that day alone. Minitaur scored 0.719, worse than a coin flip at 0.693, and Llama 1.93. Whatever Minitaur knows about these people, it learns it from what they did earlier in the session.

How much is there to learn? I wrote down a model that knows nothing about carpets or genies: the probability of taking the same carpet as yesterday, given whether yesterday paid and whether the flight was the usual one, and the probability of rubbing the same lamp as last time on that mountain, given whether it paid. Six numbers, fitted on the other 23 people and tested on the one left out. This lookup table scored 0.568. The hybrid model of model-based and model-free learning that the paper used as the cognitive baseline scored 0.607, and 0.588 when fitted to these same 24 people. A table of six numbers beats the textbook model, and Minitaur beats the table by 0.036 per choice:

[[ladder]]

For this task, then, the shortcut reading is mostly right: the instructions are worth 0.015 per choice, and the history is worth everything. The logic problems are a different kind of task ([Jansen et al., 2021](https://doi.org/10.1038/s41562-021-01057-0)). Every participant answers the same twenty questions, each with five options, and nothing in the earlier answers says much about which option a person picks next. I removed the text of the questions and options and kept only the answer letters. Minitaur went from 1.505 per answer to 1.657, worse than picking at random among five, 1.609:

[[logic]]

Here the model reads. On the question where 84% of people chose the same answer, Minitaur's score with the question was 0.88; without the question it had no idea which option that was. The fine-tuning, though, added little in this domain: the untuned Llama scored 1.523, only 0.018 worse. This fits what Oh and Gobet found across many tasks: when the content of the trials decides the choice, the models use the content.

The last test is the one the critics care most about.

The paper says Centaur can "predict and simulate", so I let Minitaur play the magic-carpet task itself: the same instructions, the reward chances of each of the 24 people's sessions, the usual flight 70% of the time, and every key press drawn from the model's probabilities for the two letters on offer (more than 99.76% of its probability was on those two letters anyway).

The classic way to read this task is to ask what a player does the day after a reward. A player who only repeats what paid takes the same carpet again whatever happened on the way. A player who understands the carpets reasons differently: if the coin came after a rare flight to the other mountain, the carpet to take next time is the other one. The model-based index measures that difference: positive when rewards after rare flights send you to the other carpet. The 24 people were strongly model-based: after a reward through the usual flight they took the same carpet 81% of the time, after a reward through a rare flight 49%, and after no reward the pattern flipped. Minitaur, playing with the same instructions, was not:

[[stays]]

Its index was +0.02, with an interval that includes zero, against +0.67 for the people. It repeated rewarded carpets a little more often, a weak model-free habit, and without the instructions it played the same way. It collected about as many coins per day as the people, 0.51 against 0.52, in a version of the task that pays little for understanding it: the untuned Llama, which barely followed rewards at all, collected 0.49.

This is Minitaur, not Centaur. The paper ran the 70B model on a spaceship version of the task and reported "trajectories in which learning is purely model-free, purely model-based and mixtures thereof", and it says Minitaur "generalizes less robustly". What the test shows is the gap Namazova and colleagues described: a model that predicts each next choice of these 24 people better than the textbook model does, given their history, and that plays the game, given only the instructions, unlike any of them.

## Prediction is not explanation

None of this means Centaur is useless. It means it is a different kind of object from what Newell asked for, and statisticians have had a name for the difference for a quarter of a century. In 2001 Leo Breiman described "two cultures": one "assumes that the data are generated by a given stochastic data model", the other "uses algorithmic models and treats the data mechanism as unknown", and in the second, "The goal is not interpretability, but accurate information" ([Breiman, 2001](https://doi.org/10.1214/ss/1009213726)). Centaur is the algorithmic culture arriving in psychology.

Psychology has seen this move before, and it went well. In 2021 Joshua Peterson, Thomas Griffiths and colleagues collected human choices for almost 10,000 gambles and found that an unconstrained neural network predicted them better than any of the classic theories. They did not stop there, because that network "provides limited psychological insight". They used it to find where theories failed, and built a mixture of theories that predicted better than any single classic theory while staying readable ([Peterson et al., 2021](https://doi.org/10.1126/science.abe2629)). Griffiths and Peterson are both co-authors of Centaur, and the paper ends with a small version of the same move: using the model to refine an interpretable account of how people combine expert ratings.

The risk is in skipping that step. Molly Crockett and Lisa Messeri argue that the ability of such models to emulate people on "Decontextualized, Engineered, Anonymized, Disembodied" tasks "says more about the limitations of the tasks than it does the capabilities of LLMs", and that relying on them risks "illusions of generalizability" ([Crockett & Messeri, 2026](https://doi.org/10.1016/j.tics.2025.09.012)).

## The authors' next step

Binz and Schulz seem to have reached a similar conclusion. In May 2026 they released Psych-201, with 208,021 participants and 25.9 million responses ([Binz et al., 2026a](https://arxiv.org/abs/2605.07632)). In October 2026 they posted CORE, "COntextual Retrieval and Encoding": "a hierarchy of episodic memories into which prediction errors are written and whose encoding, retrieval, and forgetting depend on context." It has 452,608 parameters. Fitted to 137 experiments, it predicts held-out choices "better than Centaur in 47 of them" out of 62 studies, and when simulated it reproduced "50 of the 67 effects" from the original papers ([Binz & Schulz, 2026](https://arxiv.org/abs/2610.13603)). It is a preprint, two weeks old as I write this, and nobody has replicated it yet.

The authors describe it modestly: "an intermediate step between classical cognitive models and black-box systems." That is, a model with commitments: memories, prediction errors, forgetting. Something that can be wrong in a specific way, built with the help of a model that cannot.

Read this way, Centaur did what Newell asked for in his third remedy, a single system that takes the instructions for every task, and showed why the request was not enough. A system can take the instructions and predict the next key press better than any theory, and still not play the game the way the people it predicts do. Newell wanted the theory to make the predictions. Centaur makes them, and nobody, including the model, can say why.

## References

- Newell, A. (1973). [You can't play 20 questions with nature and win: projective comments on the papers of this symposium](https://doi.org/10.1016/B978-0-12-170150-5.50012-3). In W. G. Chase (Ed.), *Visual Information Processing* (pp. 283–308). Academic Press.
- Newell, A. (1992). [Précis of Unified theories of cognition](https://doi.org/10.1017/S0140525X00069478). *Behavioral and Brain Sciences*, 15(3), 425–437.
- Breiman, L. (2001). [Statistical modeling: the two cultures](https://doi.org/10.1214/ss/1009213726). *Statistical Science*, 16(3), 199–231.
- Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., & Qin, Y. (2004). [An integrated theory of the mind](https://doi.org/10.1037/0033-295X.111.4.1036). *Psychological Review*, 111(4), 1036–1060.
- Feher da Silva, C., & Hare, T. A. (2020). [Humans primarily use model-based inference in the two-stage task](https://doi.org/10.1038/s41562-020-0905-y). *Nature Human Behaviour*, 4, 1053–1066.
- Jansen, R. A., Rafferty, A. N., & Griffiths, T. L. (2021). [A rational model of the Dunning–Kruger effect supports insensitivity to evidence in low performers](https://doi.org/10.1038/s41562-021-01057-0). *Nature Human Behaviour*, 5, 756–763.
- Peterson, J. C., Bourgin, D. D., Agrawal, M., Reichman, D., & Griffiths, T. L. (2021). [Using large-scale experiments and machine learning to discover theories of human decision-making](https://doi.org/10.1126/science.abe2629). *Science*, 372(6547), 1209–1214.
- Binz, M., et al. (2024). [Centaur: a foundation model of human cognition](https://arxiv.org/abs/2410.20268). arXiv:2410.20268, version 1.
- Binz, M., Akata, E., Bethge, M., et al. (2025). [A foundation model to predict and capture human cognition](https://doi.org/10.1038/s41586-025-09215-4). *Nature*, 644, 1002–1009.
- Helmholtz Munich (2025). [AI that thinks like us – and could help explain how we think](https://www.helmholtz-munich.de/en/newsroom/news-all/artikel/ai-that-thinks-like-us-and-could-help-explain-how-we-think). Press release, 2 July 2025.
- Bowers, J. S., Puebla, G., Thorat, S., Tsetsos, K., & Ludwig, C. J. H. (2025). [Centaur: a model without a theory](https://osf.io/preprints/psyarxiv/v9w37). PsyArXiv; version 5 (2026) retitled "On the misuse of prediction in NeuroAI: a case study of Centaur".
- Xie, H., & Zhu, J.-Q. (2025). [Centaur may have learned a shortcut that explains away psychological tasks](https://doi.org/10.31234/osf.io/u7z4t_v2). PsyArXiv.
- Namazova, S., Brondetta, A., Strittmatter, Y., Nassar, M. R., & Musslick, S. (2025). [Not yet AlphaFold for the mind: evaluating Centaur as a synthetic participant](https://arxiv.org/abs/2508.07887). arXiv:2508.07887.
- Orr, M. G., et al. (2025). [Not even wrong: on the limits of prediction as explanation in cognitive science](https://arxiv.org/abs/2510.03311). arXiv:2510.03311.
- Liu, W., & Ding, N. (2025). [Can Centaur truly simulate human cognition? The fundamental limitation of instruction understanding](https://doi.org/10.1360/nso/20250053). *National Science Open*, 5(1), 20250053.
- Crockett, M. J., & Messeri, L. (2026). [AI surrogates and illusions of generalizability in cognitive science](https://doi.org/10.1016/j.tics.2025.09.012). *Trends in Cognitive Sciences*, 30(3), 203–215.
- Binz, M., et al. (2026a). [Post-training makes large language models less human-like](https://arxiv.org/abs/2605.07632). arXiv:2605.07632.
- Oh, N., & Gobet, F. (2026). [Small foundation models of human cognition and behaviour](https://arxiv.org/abs/2608.05224). *COLM 2026*.
- Binz, M., & Schulz, E. (2026). [A small cognitive core explains human behavior at scale](https://arxiv.org/abs/2610.13603). arXiv:2610.13603.

The lab, the analysis and the figures as code: [articles/centaur](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/centaur).

{% endraw %}
