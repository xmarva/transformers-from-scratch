---
layout: post
title: "Is it rational to be good? Game theory and the ethics of cooperation"
date: 2026-10-28 12:00:00
description: "Glaucon's challenge, Hobbes's Foole and Hume's farmers read through the prisoner's dilemma: when cooperation is rational, which virtues the models select, how much forgiveness reason allows, and what game theory cannot settle"
tags: [game-theory, ethics, philosophy, cooperation, prisoners-dilemma]
categories: cognitive-science
featured: false
thumbnail: /assets/posts/game-theory-ethics/cover.png
og_image: https://xmarva.github.io/assets/posts/game-theory-ethics/cover.png
stylesheets: [/assets/css/tfs-course.css]
toc:
  beginning: true
---
{% raw %}

In the second book of Plato's *Republic*, Glaucon offers Socrates a challenge that moral philosophy has been answering ever since. People, he says, praise justice not for itself but for its consequences. Left to ourselves we would all prefer to wrong others and get away with it; we settle for justice because we fear being wronged more than we enjoy wronging. Justice, on this account, is "a mean or compromise, between the best of all, which is to do injustice and not be punished, and the worst of all, which is to suffer injustice without the power of retaliation" (*Republic* 359a) ([Plato](https://www.gutenberg.org/ebooks/1497)). To make the point he tells the story of Gyges, a shepherd who finds a ring that makes him invisible, and uses it to seduce the queen, kill the king and take the throne. Give such a ring to a just man and to an unjust one, Glaucon says, and they will end up doing the same things, since "no man can be imagined to be of such an iron nature that he would stand fast in justice", and "wherever any one thinks that he can safely be unjust, there he is unjust" (360b–c).

Glaucon's argument has a structure, and the structure has a name. Twenty-three centuries after the *Republic*, the mathematicians of game theory found it in a two-by-two table and called it the prisoner's dilemma. The table turned the old question into a precise one: under which conditions is it rational to be just, and which kind of justice is rational? This essay follows the answers that game theory gives, from Hobbes to evolutionary simulations, and marks the point where they stop. The models are small enough to compute exactly, and every picture below is computed from them.

## 1. The prisoner's dilemma

Two people can each do something for the other at a cost to themselves: keep a promise, share credit, return a favour. If both do, both are better off. If one does and the other doesn't, the one who didn't is best off of all and the one who did is worst off. If neither does, both lose what they could have had together. Game theory calls this structure the **prisoner's dilemma**, after a story Albert Tucker told in 1950 to illustrate a game that Merrill Flood and Melvin Dresher had invented a few months earlier at the RAND Corporation ([Flood, 1952](https://www.rand.org/pubs/research_memoranda/RM789-1.html)).

[[dilemma]]

Only the order of the four outcomes matters. The **temptation** to exploit beats the **reward** for mutual cooperation, which beats the **punishment** for mutual defection, which beats the **sucker's payoff**:

$$T > R > P > S, \qquad 2R > T + S.$$

The second condition says that taking turns at exploiting each other is worse than cooperating, so that cooperation, not alternation, is what the two players lose. The numbers in the table, $T = 5$, $R = 3$, $P = 1$, $S = 0$, are the ones Robert Axelrod used, and the rest of this essay uses them too.

The reasoning that follows is short and, within its assumptions, impeccable. Whatever the other player does, you do better by defecting: $5 > 3$ if they cooperate, $1 > 0$ if they don't. Defection is a **dominant strategy**, and two players who see this end up with 1 point each, while 3 each was on the table. The outcome is a Nash equilibrium, since neither can gain by changing alone, and it is worse for both than another outcome they could have reached. Individual rationality, applied correctly by both, produces a result that both regard as a failure.

This is Glaucon's compromise in miniature. Each would most like to wrong the other with impunity; each fears most to be wronged; and justice, mutual cooperation, is an agreement they would both welcome and neither has a reason to keep.

## 2. Hobbes: changing the game

Thomas Hobbes described the same structure as the condition of humanity without government. In the state of nature, where nobody can trust anybody to hold back, everyone has reason to strike first, and the result is a war of all against all, in which there is "no place for Industry; because the fruit thereof is uncertain", and the life of man is "solitary, poore, nasty, brutish, and short" (*Leviathan*, ch. 13) ([Hobbes, 1651](https://www.gutenberg.org/ebooks/3207)).

Hobbes also gave Glaucon's position its sharpest voice. "The Foole hath sayd in his heart, there is no such thing as Justice", and he argues that "to make, or not make; keep, or not keep Covenants, was not against Reason, when it conduced to ones benefit" (ch. 15). The Foole is the player who has noticed that defection dominates.

Hobbes's reply is not that the Foole makes a logical error in the single case. It is that the Foole is not in a single case. A man who breaks his covenant, and declares that he may with reason do so, "cannot be received into any Society, that unite themselves for Peace and defence, but by the errour of them that receive him", and "if he be left, or cast out of Society, he perisheth" (ch. 15). The Foole's defection is a single move, but it tells everyone what kind of player he is, and the rest of the game is played against that knowledge.

Hobbes's remedy for the war of all against all, though, is not an argument but an institution: "Covenants, without the Sword, are but Words, and of no strength to secure a man at all" (ch. 17). The sovereign does not persuade the players to cooperate; he changes their payoffs, so that defection no longer pays. In the language of game theory, Hobbes's solution is to make the game stop being a prisoner's dilemma. That works, and modern states, courts and contracts are its descendants. But it leaves Glaucon's question untouched, because it explains why people keep promises when someone watches them, which Glaucon never doubted. His question was about the ring.

## 3. Can cooperation in a single encounter be rational?

Some philosophers have argued that the dominance reasoning itself is wrong, or at least incomplete, even when the two players meet once and no sovereign watches.

The first argument is Kantian. Kant's categorical imperative reads, in Thomas Kingsmill Abbott's translation, "Act only on that maxim whereby thou canst at the same time will that it should become a universal law" (*Groundwork*, Ak 4:421) ([Kant, 1785](https://www.gutenberg.org/ebooks/5682)). One of Kant's own examples is close to the dilemma: a man borrows money with a promise he knows he will not keep. If everyone acted on that maxim, "the promise itself would become impossible, as well as the end that one might have in view in it, since no one would consider that anything was promised to him" (Ak 4:422). The defector's maxim, accept the cooperation of others and give none back, has the same shape. It works only as an exception, and made universal it destroys the cooperation it lives on.

In 1983 the cognitive scientist Douglas Hofstadter turned a related thought into an argument about the game itself. His **superrational** thinkers, "by recursive definition, include in their calculations the fact that they are in a group of superrational thinkers" ([Hofstadter, 1983](https://doi.org/10.1038/scientificamerican0683-14)). Two such players know that they will reach the same conclusion, so the only outcomes really open to them are the two on the diagonal, and of those mutual cooperation is better.

David Lewis showed that this is a special case of a deeper split in decision theory: "Prisoners' Dilemma is a Newcomb Problem—or rather, two Newcomb Problems side by side, one per prisoner" ([Lewis, 1979](https://www.jstor.org/stable/2265034)). In **Newcomb's problem**, a very reliable predictor has put a million dollars into an opaque box if it predicted that you would take only that box, and nothing if it predicted that you would also take a transparent box with a thousand dollars in it. Against a player who reasons as you do, the other player's choice is like the predictor's: your choice does not cause it, but it is very good evidence about it.

**Evidential decision theory**, which chooses the act that is the best news about the outcome, recommends taking one box, and cooperating. **Causal decision theory**, which chooses the act with the best causal consequences, recommends taking both boxes, and defecting, because nothing you do now changes what is in the box or what the other player does. Philosophers have not agreed which theory is right, and the disagreement is now half a century old.

David Gauthier took a different route ([Gauthier, 1986](https://doi.org/10.1093/0198249926.001.0001)). Rather than ask which act is rational, he asked which **disposition** it is rational to have. A *straightforward maximizer* always chooses the act with the best expected outcome; a *constrained maximizer* cooperates with others who are disposed to cooperate, and defects otherwise. If dispositions are **translucent**, if others can tell, at least sometimes, what kind of person you are, constrained maximizers are offered cooperation and straightforward maximizers are not, and the constrained ones do better. The rational choice of a disposition can then require acts that, taken one at a time, are not rational. Critics have pointed out that the argument works only to the extent that dispositions really are visible, which brings Gauthier close to the next idea: that what makes cooperation rational is that the encounter is not, in fact, single.

## 4. The shadow of the future

David Hume saw the problem in an agricultural setting (*Treatise* 3.2.5.8) ([Hume, 1739–40](https://davidhume.org/texts/t/3/2/5)):

> Your corn is ripe to-day; mine will be so to-morrow. 'Tis profitable for us both, that I shou'd labour with you to-day, and that you shou'd aid me to-morrow. I have no kindness for you, and know you have as little for me. I will not, therefore, take any pains upon your account; and should I labour with you upon my own account, in expectation of a return, I know I shou'd be disappointed, and that I shou'd in vain depend upon your gratitude. Here then I leave you to labour alone: You treat me in the same manner. The seasons change; and both of us lose our harvests for want of mutual confidence and security.

The farmers in Hume's example are not wicked; they are prudent, and prudence ruins them both. What saves them is not a sovereign but the expectation of further dealings. "Hence I learn to do a service to another, without bearing him any real kindness; because I forsee, that he will return my service, in expectation of another of the same kind" (3.2.5.9). The promise, on Hume's account, is a convention that makes this expectation public: whoever promises "subjects himself to the penalty of never being trusted again in case of failure" (3.2.5.10).

Game theory makes this precise. Suppose that after each round there is another with probability $w$, so that a relationship lasts $1/(1 - w)$ rounds on average. Now a strategy can depend on what the other player did. The simplest such strategy, **tit for tat**, cooperates first and then copies the other player's previous move. Against it, a defection buys $T$ today and costs a punishment tomorrow, and it pays to cooperate if and only if

$$w \;\ge\; \max\!\left(\frac{T - R}{T - P},\; \frac{T - R}{R - S}\right) = \frac{2}{3}.$$

The first term guards against always defecting, the second against defecting and making up in turns ([Axelrod & Hamilton, 1981](https://doi.org/10.1126/science.7466396)). Axelrod called $w$ the **shadow of the future** ([Axelrod, 1984](https://gwern.net/doc/economics/1984-axelrod-theevolutionofcooperation-ch4-theliveandletlivesysteminwwi.html)).

[[shadow]]

The most striking illustration comes from the First World War. On quiet stretches of the Western Front, British and German battalions faced each other for weeks at a time and, without negotiating anything, stopped trying to kill each other. A British staff officer was "astonished to observe German soldiers walking about within rifle range behind their own line. Our men appeared to take no notice ... These people evidently did not know there was a war on. Both sides apparently believed in the policy of 'live and let live'" ([Ashworth, 1980](https://doi.org/10.1007/978-1-349-04356-9); [Axelrod, 1984](https://gwern.net/doc/economics/1984-axelrod-theevolutionofcooperation-ch4-theliveandletlivesysteminwwi.html)). When a salvo landed by mistake, a German climbed onto his parapet and shouted: "We are very sorry about that; we hope no one was hurt. It is not our fault, it is that damned Prussian artillery." The truce ended when headquarters ordered raids, whose results it could check: "There was no effective way to pretend that a raid had been undertaken when it had not."

Two philosophical lessons follow, and the second is less comfortable than the first. The first is that the rationality of cooperation is a fact about the structure of a relationship as much as about the people in it. The same person has more reason to keep a promise to a neighbour than to a stranger at an airport.

The second is that the repeated game does not *select* cooperation; it only *permits* it. The theorems that show cooperation can be an equilibrium of a repeated game, known collectively as the folk theorem, also show that almost any pattern of behaviour can be one, including mutual defection forever and many unequal arrangements in between. Rationality alone underdetermines what rational people will do together. Something else, a convention, a norm, a shared history, has to pick one equilibrium among many, and that something is where much of morality lives.

## 5. Which character is rational?

If many strategies can sustain cooperation, which ones do well? Axelrod asked game theorists, economists, psychologists and mathematicians to submit programs and played them against each other in a round robin ([Axelrod, 1980a](https://doi.org/10.1177/002200278002400101)). The winner was "the simplest of all strategies submitted", tit for tat, sent by the psychologist Anatol Rapoport ([Axelrod & Hamilton, 1981](https://doi.org/10.1126/science.7466396)). In a second round, 62 entrants who knew this tried to beat it, and it won again ([Axelrod, 1980b](https://doi.org/10.1177/002200278002400301)). A rerun of that second tournament from Axelrod's original code with 25,000 random seeds has since found that tit for tat wins about 63% of them ([Knight et al., 2025](https://arxiv.org/abs/2510.15438)), so the result is a tendency, not a law. Here is a small tournament of the same kind, with twelve classic strategies:

[[tournament]]

The six strategies that never defect first take the top six places, and between them they win almost no individual matches. Always defect wins nine of its eleven matches and finishes ninth. The **Extortioner**, one of the zero-determinant strategies that William Press and Freeman Dyson found in 2012, is built never to earn less than its partner ([Press & Dyson, 2012](https://doi.org/10.1073/pnas.1206569109)). It loses no match and finishes eleventh. Its generous twin, built never to earn more than its partner, wins no match and finishes first ([Stewart & Plotkin, 2013](https://doi.org/10.1073/pnas.1306246110)). Axelrod's explanation is that a nice strategy can never get more than its partner in a single game, since it only defects in answer to a defection; it wins the tournament by doing reasonably well with everyone. His advice "Don't be envious" is a point about which comparison is rational: your score is measured against others in your position, not against the person across the table.

Axelrod summed up the winning character in four words, "nice, retaliatory, forgiving, and clear" ([Axelrod, 1984](https://gwern.net/doc/economics/1984-axelrod-theevolutionofcooperation-ch4-theliveandletlivesysteminwwi.html)). Each word names something a moral philosopher would recognise: benevolence, a readiness to resent injury, a willingness to let it go, and the transparency that makes a person predictable. What game theory adds is that the four belong together. Niceness without retaliation is exploited, retaliation without forgiveness turns every accident into a feud, and both without clarity are wasted, because the other side cannot learn from them.

The feud is easy to see once mistakes are allowed, as they always are in life:

[[echo]]

Between two tit for tat players, one misread gesture starts an exchange of punishments that never ends, each side certain that it is only responding. A **generous** version of tit for tat, which forgives a defection with some probability $q$, stops the echo after a few rounds ([Nowak & Sigmund, 1992](https://doi.org/10.1038/355250a0)). How large should $q$ be? Forgive too rarely, and mistakes turn into feuds; forgive too often, and a player who alternates defection and apology earns more than honest cooperators. The largest forgiveness that no exploiter can profit from is

$$q^* = \min\!\left(1 - \frac{T - R}{R - S},\; \frac{R - P}{T - P}\right) = \frac{1}{3}$$

with these payoffs ([Molander, 1985](https://doi.org/10.1177/0022002785029004004); [Nowak & Sigmund, 1992](https://doi.org/10.1038/355250a0)):

[[forgiveness]]

It is hard not to read this as a formal version of Aristotle's doctrine of the mean. "Virtue, then, is a state of character concerned with choice, lying in a mean, i.e. the mean relative to us"; to feel anger "at the right times, with reference to the right objects, towards the right people, with the right motive, and in the right way, is what is both intermediate and best" (*Nicomachean Ethics* II.6) ([Aristotle](https://classics.mit.edu/Aristotle/nicomachaen.2.ii.html)). Forgiveness here has an excess and a deficiency, and its mean is not a midpoint but depends on the situation, as Aristotle said it should: as the temptation $T$ grows relative to the reward $R$, the right amount of forgiveness falls.

## 6. Evolution and the origin of morality

If cooperative dispositions pay, natural selection and cultural imitation will spread them, and biologists have taken this as an explanation of where morality comes from. Robert Trivers opened the paper that introduced reciprocal altruism with an admission: "Models that attempt to explain altruistic behavior in terms of natural selection are models designed to take the altruism out of altruism" ([Trivers, 1971](https://doi.org/10.1086/406755)). Here is what imitation does to a population of a hundred players who copy strategies that earn more, with relationships of twenty rounds on average and one move in fifty going wrong:

[[cycles]]

With only always defect, tit for tat and unconditional cooperation available, the population never settles. Tit for tat drives out defectors; unconditional cooperators then drift in, since among reciprocators they do just as well; once there are enough of them, defectors return. Lorens Imhof, Drew Fudenberg and Martin Nowak described this cycle in finite populations ([Imhof, Fudenberg & Nowak, 2005](https://doi.org/10.1073/pnas.0502589102)). Over twenty runs of the simulation, defectors held the majority 49% of the time. When generous tit for tat is available, it takes over and holds the population: defectors held the majority 18% of the time, and the average payoff rose from 1.94 to 2.58 points per round. Unconditional kindness, on this picture, is not a stable state of a society but a phase of it, and what protects a cooperative society is a willingness to punish that erodes exactly when it seems least necessary.

The evolutionary story raises an old worry in a new form. If our moral dispositions exist because they paid, are they really moral? One answer is that the explanation of a disposition's origin is not an account of its content: a person who keeps a promise because they value honesty is honest, whatever selection did to their ancestors. Another answer comes from the way cooperation among strangers actually works. Richard Alexander argued that "moral systems are systems of indirect reciprocity" (p. 77) ([Alexander, 1987](https://doi.org/10.4324/9780203700976)): I help you, and someone else helps me, because they saw what I did. Reputation then does the work that repetition does between two people, and it requires judging others, which brings normative questions back in. Martin Nowak and Karl Sigmund showed that cooperation can be sustained by a public "image score" that rises when you help and falls when you refuse ([Nowak & Sigmund, 1998](https://doi.org/10.1038/31225)). But a player who refuses to help a known cheat then loses reputation for enforcing the norm, so a rule called **standing**, which looks at whom you refused, works much better ([Leimar & Hammerstein, 2001](https://doi.org/10.1098/rspb.2000.1573)):

[[reputation]]

When Hisashi Ohtsuki and Yoh Iwasa checked all 4096 rules of this kind, only eight kept cooperation stable, and all eight share two features: helping a good person is good and refusing one is bad; refusing a bad person is good, "because it works as sanction" ([Ohtsuki & Iwasa, 2004](https://doi.org/10.1016/j.jtbi.2004.06.005)). The evolution of cooperation selects not only for helping but for a particular moral judgement: that punishing the guilty is just and punishing the punisher is not.

Indirect reciprocity also gives an answer to Hume's **sensible knave**, who reasons: "That honesty is the best policy, may be a good general rule; but is liable to many exceptions: And he, it may, perhaps, be thought, conducts himself with most wisdom, who observes the general rule, and takes advantage of all the exceptions" (*Enquiry* 9.22) ([Hume, 1751](https://davidhume.org/texts/m/9)). Hume admitted that if a man thinks this reasoning requires an answer, "it will be a little difficult to find any, which will to him appear satisfactory and convincing" (9.23). The answer he did give was about the knave's inner life: knaves "are, in the end, the greatest dupes, and have sacrificed the invaluable enjoyment of a character, with themselves at least, for the acquisition of worthless toys and gewgaws" (9.25).

The game-theoretic answer is about his outer life. In a world of indirect reciprocity, the knave's exceptions are exactly what others observe and remember, and only a knave who could be sure of never being seen would profit from them. That knave is Gyges again, and rings like his are rare.

## 7. The stag hunt and the social contract

Not every problem of cooperation is a prisoner's dilemma. Rousseau told a story in which nobody gains by betraying anyone: "Was a deer to be taken? Every one saw that to succeed he must faithfully stand to his post; but suppose a hare to have slipped by within reach of any one of them, it is not to be doubted but he pursued it without scruple" ([Rousseau, 1755](https://www.gutenberg.org/ebooks/11136)). Brian Skyrms argued that this game deserves at least as much attention as the prisoner's dilemma: "The Stag Hunt is a story that became a game. The game is a prototype of the social contract" ([Skyrms, 2001](https://doi.org/10.2307/3218711)).

[[stag]]

In the stag hunt, mutual cooperation is an equilibrium: nobody wants to leave a successful hunt. But mutual distrust is an equilibrium too, and it is the safer one, what John Harsanyi and Reinhard Selten called **risk-dominant** ([Harsanyi & Selten, 1988](https://openlibrary.org/isbn/0262081733)). With a stag worth 4 to each hunter, a hare worth 3 and nothing for hunting the stag alone, hunting the stag pays only if more than three quarters of the others will hunt it too. The problem is not temptation but trust, and trust here is self-fulfilling: the same people, with the same values, end up in either equilibrium depending on what they expect of each other.

This is the structure of what Hume called convention, and his example is the cleanest: "Two men, who pull the oars of a boat, do it by an agreement or convention, tho' they have never given promises to each other" (*Treatise* 3.2.2.10) ([Hume, 1739–40](https://davidhume.org/texts/t/3/2/2)). David Lewis built his analysis of convention on such cases, Hume's rowers among them: a regularity is a convention if everyone conforms to it, everyone expects everyone else to conform, and everyone prefers to conform on condition that the others do (p. 42) ([Lewis, 1969](https://econjwatch.org/articles/convention-a-philosophical-studyintroduction-chapter-i-and-chapter-ii)). On this view, a good deal of what we call morality, the rules of fair division, of turn-taking, of truthfulness in ordinary talk, is less a constraint on self-interest than a way of coordinating on one of many equilibria. Skyrms noted the link with the shadow of the future: "If the probability of repetition is high enough, the stag hunting equilibrium becomes risk dominant."

## 8. What game theory cannot settle

Every model in this essay takes the payoffs as given and asks only how to play. Amartya Sen argued that this is where the theory is weakest: a person who acts only on their own preferences, with no room for commitment to others or to principles, is not a model of rationality but of a kind of impoverishment. In his words, "The purely economic man is indeed close to being a social moron" (p. 336) ([Sen, 1977](https://www.jstor.org/stable/2264946)). Experiments agree that real players are not like this. In the ultimatum game, where one player proposes how to split a sum and the other can accept or let both get nothing, the narrowly rational offer is the smallest possible one, and typical offers are 40 to 50 percent ([Güth & Kocher, 2014](https://doi.org/10.1016/j.jebo.2014.06.006)). Across fifteen small-scale societies, mean offers ranged from 26 to 58 percent, and "the higher the degree of market integration and the higher the payoffs to cooperation, the greater the level of cooperation in experimental games" ([Henrich et al., 2001](https://doi.org/10.1257/aer.91.2.73)). Even the trench truce has been questioned as an example: Andrew Gelman argued that not shooting simply lowered each side's own risk, so the soldiers may not have faced a prisoner's dilemma at all ([Gelman, 2008](https://francoangeli.it/riviste/articolo/33851)).

This sets the limit of the whole approach. Game theory can show that, given what people want, certain dispositions are rational: niceness, retaliation, a measured forgiveness, attention to reputation, a readiness to trust when trust is likely to be returned. It cannot show what people should want. The payoffs are inputs, and the question of what is worth wanting, which is most of ethics, comes before the first row of the table.

What, then, of Glaucon? Game theory answers him in part. For people who expect to meet again, who are watched by others, and who cannot be sure that their rings will keep working, justice is very often the rational choice, and the models even say which kind of justice: nice, firm, forgiving within limits, and clear. They do not answer the question Glaucon actually asked, which was whether justice is good in itself, for someone whose ring never fails. That question the models leave where Plato found it.

## References

- Plato. *Republic*, book II, 358e–360d. Translated by B. Jowett. [Project Gutenberg #1497](https://www.gutenberg.org/ebooks/1497).
- Flood, M. M. (1952). [*Some Experimental Games*](https://www.rand.org/pubs/research_memoranda/RM789-1.html). RAND Research Memorandum RM-789-1.
- Hobbes, T. (1651). *Leviathan*, chapters 13, 15 and 17 (pp. 62, 72–73 and 85 of the first edition). [Project Gutenberg #3207](https://www.gutenberg.org/ebooks/3207).
- Kant, I. (1785). *Groundwork of the Metaphysics of Morals*, Ak 4:421–422. Translated by T. K. Abbott as *Fundamental Principles of the Metaphysic of Morals*. [Project Gutenberg #5682](https://www.gutenberg.org/ebooks/5682).
- Hofstadter, D. R. (1983). [Dilemmas for superrational thinkers, leading up to a luring lottery](https://doi.org/10.1038/scientificamerican0683-14). *Scientific American*, 248(6), 14–28. Reprinted in *Metamagical Themas* (1985), ch. 30.
- Lewis, D. (1979). [Prisoners' dilemma is a Newcomb problem](https://www.jstor.org/stable/2265034). *Philosophy & Public Affairs*, 8(3), 235–240.
- Gauthier, D. (1986). [*Morals by Agreement*](https://doi.org/10.1093/0198249926.001.0001). Clarendon Press. Chapter VI.
- Hume, D. (1739–40). *A Treatise of Human Nature*, book 3, part 2, sections [2](https://davidhume.org/texts/t/3/2/2) and [5](https://davidhume.org/texts/t/3/2/5). Hume Texts Online.
- Axelrod, R., & Hamilton, W. D. (1981). [The evolution of cooperation](https://doi.org/10.1126/science.7466396). *Science*, 211(4489), 1390–1396.
- Axelrod, R. (1984). *The Evolution of Cooperation*. Basic Books. Chapter 4, [The live-and-let-live system in trench warfare in World War I](https://gwern.net/doc/economics/1984-axelrod-theevolutionofcooperation-ch4-theliveandletlivesysteminwwi.html), pp. 73–87.
- Ashworth, T. (1980). [*Trench Warfare 1914–1918: The Live and Let Live System*](https://doi.org/10.1007/978-1-349-04356-9). Macmillan.
- Axelrod, R. (1980a). [Effective choice in the prisoner's dilemma](https://doi.org/10.1177/002200278002400101). *Journal of Conflict Resolution*, 24(1), 3–25.
- Axelrod, R. (1980b). [More effective choice in the prisoner's dilemma](https://doi.org/10.1177/002200278002400301). *Journal of Conflict Resolution*, 24(3), 379–403.
- Knight, V., Campbell, O., Harper, M., Gaffney, T. J., & Glynatsi, N. E. (2025). [Reviving, reproducing, and revisiting Axelrod's second tournament](https://arxiv.org/abs/2510.15438). arXiv:2510.15438.
- Press, W. H., & Dyson, F. J. (2012). [Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent](https://doi.org/10.1073/pnas.1206569109). *PNAS*, 109(26), 10409–10413.
- Stewart, A. J., & Plotkin, J. B. (2013). [From extortion to generosity, evolution in the Iterated Prisoner's Dilemma](https://doi.org/10.1073/pnas.1306246110). *PNAS*, 110(38), 15348–15353.
- Nowak, M., & Sigmund, K. (1992). [Tit for tat in heterogeneous populations](https://doi.org/10.1038/355250a0). *Nature*, 355, 250–253.
- Molander, P. (1985). [The optimal level of generosity in a selfish, uncertain environment](https://doi.org/10.1177/0022002785029004004). *Journal of Conflict Resolution*, 29(4), 611–618.
- Aristotle. *Nicomachean Ethics*, [book II](https://classics.mit.edu/Aristotle/nicomachaen.2.ii.html). Translated by W. D. Ross.
- Trivers, R. L. (1971). [The evolution of reciprocal altruism](https://doi.org/10.1086/406755). *Quarterly Review of Biology*, 46(1), 35–57.
- Imhof, L. A., Fudenberg, D., & Nowak, M. A. (2005). [Evolutionary cycles of cooperation and defection](https://doi.org/10.1073/pnas.0502589102). *PNAS*, 102(31), 10797–10800.
- Alexander, R. D. (1987). [*The Biology of Moral Systems*](https://doi.org/10.4324/9780203700976). Aldine de Gruyter; reissued by Routledge, 2017.
- Nowak, M. A., & Sigmund, K. (1998). [Evolution of indirect reciprocity by image scoring](https://doi.org/10.1038/31225). *Nature*, 393, 573–577.
- Leimar, O., & Hammerstein, P. (2001). [Evolution of cooperation through indirect reciprocity](https://doi.org/10.1098/rspb.2000.1573). *Proceedings of the Royal Society B*, 268, 745–753.
- Ohtsuki, H., & Iwasa, Y. (2004). [How should we define goodness? Reputation dynamics in indirect reciprocity](https://doi.org/10.1016/j.jtbi.2004.06.005). *Journal of Theoretical Biology*, 231(1), 107–120.
- Hume, D. (1751). *An Enquiry concerning the Principles of Morals*, [section 9, part 2](https://davidhume.org/texts/m/9). Hume Texts Online.
- Rousseau, J.-J. (1755). [*A Discourse upon the Origin and the Foundation of the Inequality among Mankind*](https://www.gutenberg.org/ebooks/11136), part II.
- Skyrms, B. (2001). [The stag hunt](https://doi.org/10.2307/3218711). *Proceedings and Addresses of the American Philosophical Association*, 75(2), 31–41.
- Harsanyi, J. C., & Selten, R. (1988). [*A General Theory of Equilibrium Selection in Games*](https://openlibrary.org/isbn/0262081733). MIT Press.
- Lewis, D. (1969). *Convention: A Philosophical Study*. Harvard University Press. The introduction and chapters I–II in [facsimile](https://econjwatch.org/articles/convention-a-philosophical-studyintroduction-chapter-i-and-chapter-ii), *Econ Journal Watch*, 16(2), 2019.
- Sen, A. K. (1977). [Rational fools: a critique of the behavioral foundations of economic theory](https://www.jstor.org/stable/2264946). *Philosophy & Public Affairs*, 6(4), 317–344.
- Güth, W., & Kocher, M. G. (2014). [More than thirty years of ultimatum bargaining experiments](https://doi.org/10.1016/j.jebo.2014.06.006). *Journal of Economic Behavior & Organization*, 108, 396–409.
- Henrich, J., Boyd, R., Bowles, S., Camerer, C., Fehr, E., Gintis, H., & McElreath, R. (2001). [In search of homo economicus: behavioral experiments in 15 small-scale societies](https://doi.org/10.1257/aer.91.2.73). *American Economic Review*, 91(2), 73–78.
- Gelman, A. (2008). [Methodology as ideology: some comments on Robert Axelrod's *The Evolution of Cooperation*](https://francoangeli.it/riviste/articolo/33851). *QA Rivista dell'Associazione Rossi-Doria*, 2/2008, 167–176.

## Further reading

- Kuhn, S. [Prisoner's Dilemma](https://plato.stanford.edu/entries/prisoner-dilemma/). *Stanford Encyclopedia of Philosophy*.
- Hankins, K., & Vanderschraaf, P. [Game Theory and Ethics](https://plato.stanford.edu/entries/game-ethics/). *Stanford Encyclopedia of Philosophy*.
- Lloyd, S. A., & Sreedhar, S. [Hobbes's Moral and Political Philosophy](https://plato.stanford.edu/entries/hobbes-moral/). *Stanford Encyclopedia of Philosophy*.
- Cudd, A., & Eftekhari, S. [Contractarianism](https://plato.stanford.edu/entries/contractarianism/). *Stanford Encyclopedia of Philosophy*.
- Alexander, J. McK. [Evolutionary Game Theory](https://plato.stanford.edu/entries/game-evolutionary/). *Stanford Encyclopedia of Philosophy*.
- Rescorla, M. [Convention](https://plato.stanford.edu/entries/convention/). *Stanford Encyclopedia of Philosophy*.
- Skyrms, B. (1996). [*Evolution of the Social Contract*](https://doi.org/10.1017/CBO9780511806308). Cambridge University Press.
- Binmore, K. (2005). [*Natural Justice*](https://doi.org/10.1093/acprof:oso/9780195178111.001.0001). Oxford University Press.

The models, the simulations and the figures: [articles/game_theory](https://github.com/xmarva/transformers-from-scratch/tree/main/articles/game_theory).

{% endraw %}
