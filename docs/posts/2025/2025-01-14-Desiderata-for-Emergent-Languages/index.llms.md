> \[coordinate decision making\], this is a difficult problem, because:  
> Firstly, most coordination problems cannot be solved in polynomial time.  
> Secondly, it’s difficult because as we decentralized our systems we need to know who, what, and when we want to communicate, right?  
> And thirdly, it’s difficult because it’s not at all clear what strategy we want to follow when things don’t actually work as expected  
> – ([Today 2021](#ref-RoboticsToday_2021)) Amanda Prorok

## Introduction

This working paper aims to **collect and develop** desiderata for emergent languages. Inspired by the reductionist approach taken by ([Skyrms 2010](#ref-Skyrms2010signals)), my investigations of emergent languages began with minimal extensions to the Lewis Signaling Game.

In a a companion working paper, “[The Many Path To A Signaling System](../2025-01-05-Many-paths-to-signaling/index.qmd)” I investigate different setting under which emergent languages can arise. These two papers help cement my intuitions on how one may develop agents with the ability to learn emergent languages with such desirable properties. I am putting this draft out to aid other researchers in the field of emergent languages as well as to solicit feedback on the desiderata I have compiled.

> **TIP:**
>
> Since writing the first draft I also took an online course on Multi-Lingual NLP.
>
> In the course the instructors suggest that if there is no data for the language we can use a related language to bootstrap the process and if that is not available they suggest using Turkish. One issue raised is that having one such language can lead to overfitting to that language. Some NLP models can be fit on multiple languages allowing them to fit to the structure of the languages rather than one specific language.
>
> One of my main takeaways from that course is that we may be able to evolve a variety of emergent languages that are approximations of Low Resource Languages. This may be possible by imbuing them with properties drawn from sources like [WALS](https://wals.info/) the World Atlas of Language Structures, c.f ([Dryer and Haspelmath 2013](#ref-wals)). Such emergent languages may then be useful as a priors for building model for transfer learning between high resource languages and the low resource ones. They may be used with prototype of NLP models to assist collection of actual data most efficiently from speakers of these languages.

> **TIP:**
>
> [![Emergent Languages in a Nutshell](../../../images/in_the_nut_shell_coach_retouched.jpg)](../../../images/in_the_nut_shell_coach_retouched.jpg "Emergent Languages in a Nutshell")
>
> Emergent Languages in a Nutshell
>
> This is a list of properties that I and others have suggested for emergent languages.

([Skyrms 2010](#ref-Skyrms2010signals)) and ([Foerster et al. 2016](#ref-NIPS2016_c7635bfd)) have shown that we can also add a lot of structure to the signaling system.

In his book he considers, learning to reason, learning to form efficient communication networks, learning to use better communication protocols. Others have considers inducting communication protocols from a meta protocol. ([Foerster et al. 2016](#ref-NIPS2016_c7635bfd)) And this has been coupled with the idea of learning to represent the structure of the state space.

I think that **the coordination problem is the easy part** and I have devoted some time to solve it in a number of settings. I think that the next challenge is to learn a reductionist concept for signal aggregation. Something that allows bags of words, ordered sequences and recursive structures to be learned in much the same way. I think this we have not been asking the right questions to find it yet since aggregation is a hard problem. However I think that at this point a starting point exists and the next problem is to learn the structure of the state space. I think that here we can apply some existing algorithms that may be a good fit for Lewis signaling games but it is quite important because the structure of the state space is what will determine the optimality of the language to a particular domain. A second facet is that we should think of a minimal state space that captures the essence of the real world. This should be a model that we can consider sufficiently powerful for real world agents. But in terms of the emergent signaling system I don’t think it will be qualitative or quantitatively all that different from other systems other than this it could act as a good [interlingua]() for transferring between natural languages as well as between tasks in RL. This is therefore the holy grail of emergent languages at this point.

## The Desiderata

So the desiderata for emergent languages are

Important:

## Learnability – Easy to learn

Despite its importance, learnability may be overlooked in research on emergent languages. While some settings may not prioritize ease of learning, in lifelong learning environments or multi-generational agent interactions, learnability becomes a crucial factor. Many other desiderata either enhance or hinder learnability

> My experience with the Lewis signaling game is that it is easy to learn and that Natural Languages are not. The difficulty seems to be in finding the right structure for the state space so that the signaling systems generalize well, allowing learners to pick up the language from a few examples and start to communicate effectively with a limited proficiency. At a deeper level if the language arises though a mechanism of spontaneous symmetry breaking, i.e. random choices then will have long term impact on the emergent languages that will increasingly be harder to undo.

### Insights from Natural Languages

Children typically begin learning to talk in stages:

| Age | Approximate Vocabulary Size | Language Milestones |
|----|----|----|
| 0-6 months | rudimentary signaling | Crying, cooing, babbling |
| 12 months | ~1-5 words | Babbling becomes more complex, First words emerge (e.g., “mama,” “dada”) |
| 18 months | ~50 words | Vocabulary expands, simple word combinations begin |
| 24 months | ~200-300 words | Two-word phrases, expanding vocabulary |
| 30 months | ~400-600 words | Three-word sentences, basic grammar structures appear, more verbs/adjectives |
| 36 months | ~1,000 words | Complex sentences, basic grammar |
| 4 years | ~1,500-2,000 words | Storytelling, past/future tense |
| 5 years | ~2,500-3,000 words | Longer conversations, complex grammar |
| 6 years | ~5,000+ words | Adult-like speech patterns emerging |

Language development varies by individual, but most children can hold simple conversations by age 3 and have a well-developed vocabulary by age 5.

| **Natural Languages**          | **Emergent Languages**         |
|--------------------------------|--------------------------------|
| proceeds in stages             | has one stage                  |
| takes about 5 years            | takes o(n^2/2) steps           |
| in 5 years 5000 words          | 2,500,000 time steps[^1]       |
| Notoriously difficult to learn | Easy to learnO(N^2/2) [^2]     |
| learning in stages             | learning in one stage          |
| learning is a lifelong process | learning is a one time process |
| learning is a social process   | learning is one on one         |

Natural languages are notoriously difficult to learn.\| On the other hand emergent languages can be learned in the worst case as quickly as in O(N^2/2), that is if one neutralizes the stochasticity of the process by requiring that nature prioritize unlearned states before all the learned ones at any given time.\|

It doesn’t seem to be much easier to learn a second language once you have already learned the first. However research treats first language acquisition as a different process to second, etc. language acquisition.

In terms of the Lewis signaling game the first language acquisition maps the signals to states, a second language maps the signals to the first language. However in the game the players also engage in the inventing of the language for the first time simultaneously with learning about the state of the world.

The exception seems to be that children master one by the time they are 5 years old. Over time they will improve their proficiency and may learn additional languages. Students of a language may require many examples to learn. Having a dictionary is of limited help. [^3] There isn’t an objective metric for tracking how difficult it is to learn a particular language. There is plenty of anecdotal evidence that some languages are easier to learn, as well as some languages are harder for native speakers of a second language. There are many challenges along the way.

Part of [Nature v.s. Nurture debate](https://en.wikipedia.org/wiki/Nature_versus_nurture) is to what degree is the language instinct hard coded into us. c.f. Pinker’s books the blank slate and the language instinct. It

I often get hung up on why are natural languages (AKA man’s greatest invention) are such a challenge to learn and what this might mean for my investigation into emergence of language.

Hypothesis: Complex signaling that fulfill enough desiderata may suffer from reduced learnability. I think that the core desiderata might actually allow for languages with graded burden of learnability.

Questions: Howe can we evaluate the learnability of a signaling system? What are the metrics that we can use to evaluate the learnability of a signaling system?

#### Metrics for Learnability

##### TODO:

I’m uncertain if other have studied learnability in the context of emergent languages. So there is an open challenge in defining good metrics that can guide progress in this area.

research this in the litrature.

In RL we do however have metrics that are associated with learning. These are: 1. The number of examples needed to learn a task. 2. The generalization of the learned task. 3. The stability of the learned task.

1.  Evidence that agents have learned the signaling systems is that they can communicate effectively. This is measures in terms of expected success rate. During learning this may take a long time to converge to 1.0.

2.  In the tabular setting n^2/s seems to be the worst case for agents engaged purely with learning a signaling system.

3.  The best case is O(n) - i.e. if the receiver could see the state it would still need to see The faster the agent reach 1.0 the better they

4.  The most obvious metric is the success rate for the agents in carrying out communications. However this by itself is not enough. In the original game the best case path to learning generally requires at least n^2/2 tries and the algorithms used do not usually need to generalize. In the complex signaling systems we might have infinite or prohibitively large state spaces and even for finite one potentially unbounded number of signals.

5.  A second idea is how many examples are needed to be seen before agent get a good grasp of the language.

- In ([Goldsmith 2001](#ref-goldsmith-2001-unsupervised)) the author considered the induction of morphology using Minimal description length MDL - the number of bits needed to describe the signaling system is what agents need to coordinate between them to learn a shared communication system. In Goldsmith’s work he considered a corpus and then compared it to to the a corpus compressed + the binary of an encoder built from templates and a frequency lexicon. Goldsmith showed that by learning a morphology it was possible to compressed corpus + the binary of the encoder into less then the original corpur. his is a good metric for the learnability of a signaling system. In the case of emergent languages

- I should make a review of this great paper but the gist is that there is a lexicon and a morphology based on a set of templates that are used to generate the words. [^4]

- We like to consider two cases: [^5]

1.  there is a sender/teacher with a good signaling system and a receiver/student learning it.
2.  there is no sender/teacher and the agents have to construct such a signaling system from scratch.

the above notion if MDL is a good metric for the first case but not the second. In the second case we need to consider the complexity of the state space as well as the algorithmic complexity arriving at a common communication system. The cost of coordination of an MDL is subsumed by the cost due to complexity of constructing optimal signaling system that faithfully represent the structure of the state space.

Another two points:

- Learning a partial system should give agents better benefits than not.
- Learning as a group should be easier and quicker than learning individually.
- e.g. Learning of rules (grammar/morphology) should amplify the learning and generalization of the speaker wrt the structure of the state space.

RL based metrics for learnability are:

1.  cumulative reward G_T if normalized becomes the average reward per time step or expected success rate.
2.  Sample efficiency
3.  convergence rate - how long till policy stabilizes or stops improving
4.  regret - Measures how much reward is lost due to suboptimal actions compared to an optimal policy. Defined as R_t = G^\*-G_t where G^\* is the optimal cumulative reward
5.  entropy of the policy- Measures randomness in action selection
6.  policy stability
7.  success rate/ task completion rate
8.  generalization
9.  exploration vs exploitation ballance

For hierarchical RL additional metrics may include subtask efficiency, hierarchical consistency, and intrinsic reward utilization to assess the learning of macro-actions and task decomposition.

#### Gold’s Theorem

One of the main points in the paper is Gold’s Theorem, c.f.([Gold 1967](#ref-gold1967language)) concerning the impossibility of learning an unrestricted set of languages. The authors also discuss the necessity of innate expectations in language acquisition, arguing that the human brain’s learning algorithm can learn existing human languages but not all computable languages.

> Gold’s theorem formally states there exists no algorithm that can learn a set of ‘super-finite’ languages. Such a set includes all finite languages and at least one infinite language. Intuitively, if the learner infers that the target language is an infinite language, whereas the actual target language is a finite language that is contained in the infinite language, then the learner will not encounter any contradicting evidence, and will never converge onto the correct language. This result holds in greatest possible generality: ‘algorithm’ here includes any function from text to language.

#### The paradox of language acquisition.

The authors describe Chomsky’s concept of “poverty of the stimulus” and the proposed solution of “universal grammar” (UG) as a restricted set of candidate grammars. They explain the controversy surrounding the notion of an innate UG and highlight the role of learning theory in demonstrating its logical necessity.

If memory serves me correctly, in ([Bod et al. 2003](#ref-bod2003probabilistic)), the authors argue that the `poverty of the stimulus` is not a problem for language acquisition. They show that if Grammars are defined using probabilistic rules then the poverty of the stimulus is not a problem. This is because the child can learn the language by observing the world and the language spoken by the adults. T think that they make a case against the necessity for an innate UG.

I have not thought everything through but I believe that Gold Theorem may not be a show stopper for learning grammars in MARL.

- One reason is that saliency and other factors may restrict the sets of all possible languages that may be learned to just one.
- A second is that spontaneous symmetry breaking can also reduce the number of possible languages to just one.
- I cannot however say this is a general refutation of Gold’s Theorem, I think that it depends of the choice of State space and it’s implicit structure and how that might be captured in terms of a signaling system.
- If
- Another is that One reasons is my view that bayesian agents can learn to update thier beliefs about the relevant equilibria.

This is another point they make that I wanted to bear upon. In the book probabilistic linguistic the authors if agents can learn an approximation of a language then they can learn the language. This is a point that I have made in my own work.

### References

In the beautiful review paper ([Nowak et al. 2002](#ref-Nowak2002ComputationalAE)), the authors discuss Learnability. This shows that this at least in the evolutionary context this has been considered

I also drew the following sources for this section:

([Bloom 2013](#ref-bloom2013one)), ([Fenson 2000](#ref-fenson2000variability)), ([Hoff 2009](#ref-hoff2009language)), ([Hart and Risley 1995](#ref-hart1995meaningful))

### Optimal for Communication

Agents should achieve a high success rate in communication. To better understand signaling and complex signaling systems such as those that emerge out of MARL agent interaction it is necessitate to think in terms of a information-theoretic formulation of the problem. The Lewis signaling game is very simple and lacks any assistance in this regard. However

Agents should be able to communicate with a high success rate. (This is a doorway to information theoretic formulations)

Emergent Communications should have an expected success rate of almost 1.

Many systems with with expected success rate less then are acceptable however we can tend to see agents reach close to 1.

For optimal communication across a noisy channel we need to consider the following:

Prefix codes are optimal for communication. A prefix code is a type of code system (typically a variable-length code) distinguished by its possession of the “prefix property”, which requires that there is no whole code word in the system that is a prefix (initial segment) of any other code word in the system. Adaptive ones like the [Vitter algorithm](https://en.wikipedia.org/wiki/Adaptive_Huffman_coding) appear to be candidates that play well with many of the other desiderata like: - brevity (prefix codes as source coding) - resilience to errors (channel coding) and adding error detection and correction codes to the message. - easy to decode (minimal decoding complexity) - able to adapt to new states and changes in the distribution of states. - distributional semantics. - compositionality. - zipfian distributions.

Another more powerful code is arithmatic coding, however this is not practical for a number of reasons.

## Resilience - to Errors

The work of … suggests that

### Overview

Signaling systems should be resilient to errors. Errors can be introduced in a number of ways: - Noise in the channel. - Errors in the encoding or decoding process. - Complete and Partial pooling of equilibria.

As we inject errors into the signaling system we should see a number of features from natural languages emerge.

### Insights from Natural Language

Natural language have a number of features that make them resilient to errors:

- Redundancy.
- Distributional semantics.
- Agreement.
- Vowel harmony.

An likely other I am not aware of.

IN ([Nowak and Krakauer 1999](#ref-nowak1999evolution)) the authors suggests that the evolution of language is driven by the need to be able to communicate effectively in the presence of errors. They demonstrate that the number of vowels and consonants might increase to a goldilocks zone but beyond a certain threshold the the number of errors will increase making a larger phologoical inventory less likely to evolve.

### Metrics

This property may be measured by the number of errors that can be introduced into the signaling system before it expected success rate drops below a certain threshold.

### Resources

This is based on …

## Decodeability - Easy to decode

### Overview

Complex signaling systems should be easy to decode. This is a property of prefix codes. Huffman coding has this property.

It is worthwhile considering huffman coding as a working model for the evolution of signaling systems.

\[\] Todo: insert reference to unwritten article and code on n-arry huffman coding of syllable based signaling systems.

### Insights from Natural Language

Are natural languages easy to decode? Parsing natural language is a difficult task is challenging for machines but not so much for humans.

### Metrics

Ease of decoding is likely an information theoretic property and should have metrics that are associated with the complexity of the decoding process.

### Resources

This is based on ([MacKay 2003](#ref-MacKay2003) ch. 5 p.91-6)

## Salient wrt. the distribution of states

Salience means that some equilibria are better than others in terms of some practical concern that affect the fitness of the signaling agents.

e.g. a log-normal or exponential distribution or any distribution other than the uniform distribution of states will have a different optimal signaling system, particularly under (adaptive) prefix coding.

e.g. is signals have differential costs this will create a parallel mechanism which I consider to be a form of salience. More specifically, if agents use a binary systems with high and low frequency signals. And the low frequency signals are harder for predators to detect then the agents should reduce the number of high frequency signals for predators signaling. On the other hand using a long sequence of low frequency signals will require longer signals and thus also increases the risk of being predation during the signaling process.

This suggest that out of the numerous signaling systems that can arise from the Lewis signaling game some will be more efficient and this may well translate directly into greater fitness for the agents.

However we might as well be clear that for larger state spaces the signals further down the long tail play a less important role in the fitness of agents and therefore one a signaling system or even a partial pooling equilibria is found that captures sufficient quantiles of the state distribution, may confer similar fitness benefits.

And so it seems that while saliency places an order on all possible signaling equilibria it may have to compete with less salient equilibria that are much easier to learn and use. H

However I think that learnability demands much more on the structure of the state space in terms of prelinguistic objects while saliency is more about the distribution of states. So that if we try to disregard most pathological cases and consider state spaces resembling reality we might expect that the most salient signaling system will also be very learnable.

e.g. is syllables have energy/clarity costs or then the otherwise symmetrical equilibria will now have an order and there is a notion of the most salient signaling system.

Salience is not the same but may also encompass the notions of

- risk minimizing wrt risks associated with signaling - particularly in the case of risks affecting agent’s fitness!
- minimize costs/overhead associated with signaling (in rl there should be a cost associated with each marginal bit that the they send across the channel)
- Minimizing signal length - This may be the reason why the most common states are the shortest signals - using the unmarked case as the default. This is a form of source coding. (Perhaps this items is more fundamental then risk and salience) It may also be the reason why we have vowel harmony in some languages and why there are other types of redundant agreement in different languages.
- A theorem: if a (natural) language arising via evolution has a redundancy that may be removed without loss of information or via context then it will be compressed and eroded or eliminated given time. Thus such features are that exist and are stable are will have a measurable benefits in terms of communication.

### Diminishing significance of salience

A final thought on salience. We considered a couple of examples like riskiness of predators and their frequency of occurrence. However many other factors may arise that contribute to salience and they may be due to cross-cutting concerns and create a pareto frontier of equilibria. These may be too complex for the agents to process or to plan for.

Once a language is established in a large enough a population it would become impractical to coordinate all speakers to a equilibria with a greater salience.

salience will be ignored when: \text{Cost of new-coordination in the population} \> \text{Benefits of salience}

This is a form of the principle of least effort. But I doubt we can formalize this in a way that is useful as coordination costs are algorithmic while fitness an emergent property of fitness and depends on the different aspects for salience. These cost are generally incompatible units and thus a challenge to reconcile.

## The signaling system should be able to generalize to new states

In its simplest for generalization is due to

1.  having reusable templates for recurring patterns
2.  having predictable semantics from those templates

i.e. if sender just picks all the 6 syllables sequences from a set of 10 symbols. Then are a million possible signals and assigns them arbitrarily to a meaning there is no generalization. Learning would require 10^12/2 steps.

\<\text{s-noun} , \text{declension} \> \<\text{verb} , \text{conjugation}\> \<\text{o-noun} , \text{declension} \> \qquad \text{three word template} \tag{1}

The above template might be used for SVO language with the same six syllables. This three word template is about the level mastered by a 3 year old.

There agent need to learn 10 subject and 10 subject declensions, 10 verbs and 10 verb conjugations. And 10 objects and 10 object declensions. This is a total of 60 basic symbols that require 1800 steps to learn. But these give rise to a system that is highly predictable. More so if the same noun and declension is used for the subject and object where the learning is only 800 steps.

Three year olds typically have a larger vocabulary than 10 nouns and 10 verbs. But the point is that once this system is established the agent can expand the vocabulary of nouns and verbs without having to learn the declensions and conjugations again and can use to generate vast amount of new sentences with little effort in terms of learning.

### Contextual semantics and generalization

Now I want to point out that while this generlises nicely we have a finite number of words and sentences from this system. This is essentially a tabular lexicon. Also we do not consider the use of context. i.e. 

\<aa-ba\> \<ka-da\> \<ba-ra\>

\<aa-ba\> \<da-ra\> \<ba-ra\>

the nouns in the first sentence are assumed to mean the same things as the nouns in the second sentence.

If the verbs were such that this semantics was incompatible, then we might assign different meanings to one or both nouns. This would however require more learning steps.

Note: that in this case \<aa-ba\> has two meanings \<aa-ba/1\> \<aa-ba/2\> and we need to pick the correct one using the verb. Thus there is a tradeoff between the number of meanings and the number of learning steps and the use of context for semantic disambiguation.

So the problem isn’t so much in terms of the learning but it is in terms of making sure we don’t create collisions between the different semantics. \<aa-ba/1\> \<ka-da\> \<ba-ra\>

\<aa-ba/2\> \<ka-da\> \<ba-ra\>

We have to be sure there aren’t some combination like the above that both make sense as we would not be able to tell which one was meant. I.e. we end with a partial pooling equilibrium.

An easier path to learning perhaps is to add more one word syllable nouns and verbs. Nowak and Krakauer (1999) suggest though that after some point increasing the inventory of sounds leads to more errors in communication.

At this point it might be easier to allow a second syllable to the noun slot. Again we run into some problems, we reused the same syllable for the noun and declension, so we can’t be certain if a triplet is a long noun with a declension or a short noun with a declension and a verb. Again we might be able to figure this out from the context but can we be sure there isn’t a collision.

On way to dix this is to use a stress marker on the last sound of a verb or noun. This would be the penultimate sylable of the inflected form.

\<aa'-ba\> \<ka-da\> \<ba-ra\>

\<aa-ba'-ka\> \<da-ba'-ra\> \<ba'-ra\>

This allows us to have open categories of nouns and verbs.

Another way to move forward is to have more templates that further increase semantics in a predictable ways.

For example we might add adjective and adverbs.

We can add up to 10 grammatical gender to a noun and assign each to some semantic category. This can allow us to use more verbs in a more predictable way.

For verbs we can enhance our 10 primative verbs with 10 coverbs to and have 100 derived verbs. Some might mark aspect and mood and become attached to the conjugation.

So we have tabular systems

\<\text{gender}, \text{s-noun'}, \text{declension} \> \<\text{verb'} , \text{conjugation}, \text{co-verb}\> \< \text{gender}, \text{o-noun}, \text{declension} \> \qquad \text{template with gender and coverbs} \tag{2}

\<\text{adj}, \text{s-noun'}, \text{declension} \> \<\text{verb'} , \text{conjugation}, \text{adverb} \> \< \text{adjective}, \text{o-noun}, \text{declension} \> \qquad \text{template with adverbs and adjectives} \tag{3}

Ok so we have a system that is highly predictable, extendable and yet easy to learn. If the gender, co-verb, adjective and adverb are unstressed the agent should be able to make sense of the multiple templates with minimal collisions and we might even use a prefix to indicate the template being used.

An easier method might be to break the big templates into smaller ones. However this will be covered in the next section as it would complicate things in a way that might be unnecessary.

## Hierarchical structures

Hierarchical structures arise naturally out of rule based systems. Rules lead to trees structures - which are ubiquitous in language.

Now we already laid the ground work for this in the previous section. We created many tables using a few templates. If we replace the templates with rules we are one step away from a grammar.

What we are missing is a recursive rule. A recursive rule allows us to create a structure that is no longer infinite.

A recursive rule is one that can be applied to itself. This is a very powerful concept. It allows us to create structures that are infinitely deep by repeating the same rule over and over again.

Another way to think about this is that a rule is recursive if it a;lows to replace a symbol with a copy of itself. Alternatively it might be replaced by a parent of itself.

S \rightarrow aSb \tag{4}

## an example or recursive rules with noun phrases.

\$\$ \begin{align\*} S &\rightarrow NP VP \\ NP &\rightarrow (Det) N \| \\ VP &\rightarrow V (NP) (PP) \end{align\*}

\begin{align\*} S &\rightarrow a\|b \\ a &\rightarrow bb \\ b &\rightarrow aa \end{align\*}

I think this is an integral part of generalization and of learnability. I think that this is a grammar that can generate nested structures like nested clauses, nested phrases, or nested dependency trees.

A Recursive grammar seems more compact than one that is not recursive. It is also more expressive. Recursive rules are more learnable in the sense that learning one rule leads to possibly infinite number of sentences.

Another property is that the recursive grammar should be well formed (syntax) and avoid generating ungrammatical sentences (semantics) This may be complicated and I’d rather avoid hashing the details at this time. If necessary this item be broken down.

This is nice to have because a grammar with non recursive rules is less expressive. We might run into problems for expressive basic recursive structures like numbers.

- every time an agents learns another part of the system it should have. My solution here leans on using group actions to structure the state space. Either one big one group action like for hebrew or a number of smaller ones like for english.

I’m not sure if this should be included. Generalization implies two or three things

1.  replacing tabular lexicons with linear function approximators

2.  compositionality over multiple analogies.

3.  Recursive grammar. This is one needs to be promoted or at least combined with learnability. Generalization means beinng able to learn base forms and generelizing to derived forms autoamtical. It also means being able to compose signals. IT also being able to capture hierarchies and other structures in the state space.

4.  Morphology

    - inflections-
      - verb - conjugations
        - Tense and Aspect and Mood systems
      - nouns - declensions
    - derivations to other parts of speech
    - paradigms for each (e.g. posseives in English)

5.  Relations

    - Sematic Roles
    - Coverbs and Complements extend the verbs to cover many related concepts

6.  Syntax for Logical and set theory as a basis for

    - Conjunctions and Disjunctions,
    - First order/Contrafactual logic Quantifiers
    - Subordination and Coordination

7.  Spatio-temporal events

    - Adverbs and Prepositions
    - Adjectives and Adverbs

## Distributional Semantics

This central property does not arise in many signaling systems. However distributional semantics is a property of natural languages. It is the good part of entanglement and disentanglement. It makes it easier to understand things by thier context[^6] & Distributed Representations

- Signaling systems should be alignable with 2000 discourse atoms c.f. ([Arora et al. 2018](#ref-arora2018linearalgebraicstructureword)), or a subset if they come from a much simpler state structure.
- In fact a major point to research on emergent languages get to see if they manifest distributional semantics. I hypothesize that this will happen if the the state space is has a semantic basis - i.e. the state space is a vector space with dimensions that are semantically orthogonal. I now revised the above strong requirements to the following but perhaps they are even equivalent.
- prefix codes with compositionality.
- prefix codes learned with loss analogies.

## Domain Specific Language (DSL) and General Purpose Language (GPL)

The languages that tend to emerge in one MDP would tend to be domain specific. However if agents experiences span multiple MDPs, their language may become more general purpose.

The DSL should encompass the states and action spaces of a particular MDP.

The GPL should have prelingistic objects that can be mapped to and from the states and action spaces of the MDPs.

### Theories of Mind and Grounding

The agents need a DSL to think strategically about its own actions in terms of the actions of others agents. This is the domain of game theory. However the linguistic abstraction for this can be considered a theory of mind. The theory of mind is in some sense a mapping between the GPL and a specific DSL. It allows the agent to think in more general terms about the MDP it is in.

The process of creating such a mapping in sometimes called grounding. It is a process of identifying states, actions, rewards and higher order abstractions of a DSL for an MDP into a GPL.

For example even if the agent has learned some Options and Generalized Value Functions (GVFs) in previous MDPs, these may not be directly applicable to the current MDP. But if the agent has a theory of mind, it may be able to map the options and GVFs to the current MDP using its DSL. Its fairly clear that a option might transfer more readily while a GVF might require learning from scratch. However many concepts from Chess and Go and Black Jack might be directly applicable to other games or they may be need to be relearned in the new MDP. However a theory of mind should allow the agent to create hypotheses about such options and GVFs from the first game. It problem might be to identify what GVF best embodies ideas like the center or tempo and pawn structure in chess and or the concept of a bluff in poker and so on.

### Inductive Bias and Theories of Mind

To empower the agents to create a DSL with the GPL in a few shot fashion it would need to test many hypotheses and establish policies and or value functions for the current MDP but based on the DSL of the previous MDP.

These hypotheses might be derived using a bayesian approach to preexisting inductive biases. Simple models would seem to first first. More complex models would better at explain as more experience is gained. A hierarchical approach might allow it to assemble more complex models from pieces of existing models. This is one advantage of a bayesian approach.

Mapping the GSL and DSL

Pehaps also to operate in terms of temporal abstractions like options and generalized value functions that may be less obvious in the current MDP.

The mapping may be viewed as embedding the agents theory of mind and the process for this is grounding.

One way for an agents to quickly learn more general purpose language is to perhaps equip them with priors that embody and inductive bias towards the general purpose language. I.e. if they have access to priors that embody different theories of mind they can use baysian occums razor to pick the one most compatible with thier experience so far.

This should let them find a GPL that is compatible with the current DSL for the MDP they are in. If they have access to temporal abstractions like options, they might repurpose then once they use the theory of mind to assign a symbolic mapping to the options’s preconditions and post conditions in terms of the current language.

### Inductive Bias and Theories of Mind

The general purpose language encompasses the DSLs of specific mdps.

some ideas of what a general purpose language should have that may not be in a DSL:

A signaling system A Domain specific language

- arithmetic or at least a number systems

- basic relations e.g. equality, and order relations.

- physics or at least spacio-temporal events

- semantic hierarchies aka thesauri

- first order logic + modal + contrafactual logic or at least propositional logic

- set theory

- basic notions from probability.

- its worth noting that set theory or logic might be all we need to get the rest, but we are not building mathematics but a language that needs to communicate.

## 9. compositionality - is state has structure the languages it should be preserved/mirrored by the language

significant increase in what it can express and understand. I lean towards adding a topology to captures semantic hierarchies. The different signaling system are associated with a lattice of topological groups with the complete pooling equilibrium at the top and the unstructured separating equilibrium. In between are partial pooling equilibria and the various structured separating equilibria. For compositionality we want to pick certain structured pooling equilibria over the structured separating ones.

### 10. Disentanglement

- we like most morphmes to have unique semantics and not require context to disambiguate them. This places too much of a congnitive burden on speakers and requires learning by learners.

### 11. Entanglement

- the language should be able to encode multiple concepts in a single signal when binding moephems etc is clearly more efficent (we never use parts in isolation)

- when language encode two or more semantics in a single signal. e.g. ‘They’ encodes (third) person and plural (number) as one signal. This is a pronoun but it is not inflected and is not made of two bound morphemes, it is a single morpheme.

I want to come up with a information theoretic notions behind driving Entanglement and Disentanglement. 1. I think they are based on the mutual information between the signals and the states and relative entropy. 2. THe number of sub-states in the structure is high it best encoded as a group action i.e. a rule 3. If the sub-states are a few it is best encoded as a dictionary 4. If like a pronoun a complex signal is high frequency and high entropy there is a greater fitness to compress them into a single signal. And we might want to reduce errors by intentionally boosting the \[phonemic contrast\][^7].

In reality natural languages are not optimal in any of these desiderata. They are the result of a long evolutionary process that has been shaped by many factors. However I think that the desiderata are a good starting point for designing a language that is optimal for a particular task.

## 12. stability of regularity and irregularity (resilience to errors and to evolution)

consider that a language that generates entangled structures to compress and reduce mistakes for something like its pronouns these should be stable over and not replaced by a more regular system that is less efficient…. i.e. the loss for having such a pronouns should be less then a the gain from having a more regular system.

Morpho-syntax should be stable over time and be composable with the abstract morphology structure of the state space.

Languages change over time but not the core structure of the language. This is a form of stability.

## 13. Zipfian distributions

## An evolving desiderata of Emergent Languages

1.  mappings between states and signals

    1.  morpho-syntax mappings preserve partial states (Homomorphism of normal subgroups)
    2.  mappings preserve semantic topologies (if a is close to b then f(a) should be close to f(b))  

2.  Stability

    1.  Regularity is stable (mappings used in syntax and morphology are stable over time)
    2.  Irregularity is stable (mappings used in irregular verbs and nouns are also stable over time) In english we maintain many irregular borrowings from other languages and thus preserve thier regularity - making such exceptions easier to learn too.

3.  Compositionality

4.  Brevity (source coding)

5.  Self correcting (channel coding to detect errors and correction them through redundancies like agreement, vowel harmony, etc.)

6.  Learnability - how many things need to be coordinated; the complexity of the structures, the [Hoffding bound](https://en.wikipedia.org/wiki/Hoeffding%27s_inequality) on rates of learning distribution when there are errors. The [Bonferroni correction](https://en.wikipedia.org/wiki/Bonferroni_correction) for multiple learners.[^8]

7.  Stable irregularities

8.  Zipfian distributions - the most common signals should be the shortest and the least common the longest. This is a form of source coding and would arrise naturally from huffman coding, except that this isn’t practical for several reasons. It could also arise out of laziness in the sender

9.  Faithfulness

10. Distributional stability

11. Decidebility - easy to disambiguate ambiguous signals from thier context

12. Expressivity - the ability to express a wide range of concepts

13. Generalization - learning the language is possible from just a few signal state pairs.

Some metrics

1.  Compositionality
    1.  Topographic similarity
    2.  
2.  Source coding
    1.  Compression ratio
    2.  Entropy
    3.  Mutual information
3.  Error detection and correction
    1.  Error rate
    2.  Hamming distance
4.  Learnability
    1.  Number of examples to learn a new word
    2.  Number of examples to learn a new rule

Another random thought or two:

## Language that can evolve under changing conditions without losing the core structure

- this is being open to extension and to change.

- I have already demonstrated above how a language can use a simple template yet be still be open to extension using open categories of nouns and verbs by applying a stress pattern to the final syllable.

- I am thinking about a stronger form of this property akin to what might be used by agents that undergoes metamorphosis and now perceives a new MDP with different states and actions. There may be a need to adapt the language and memories for the states of the world in terms of the old language. If the agent is now color blind it may benefit in the short run from a language that lacks colors. If however it regains it sight it would be better if these words were not lost.

So we might want to preserve the old semantics using new terms. This suggest that we may want to have a general purpose language but only use a small easy to learn subset of it to get going.

# Desiderata from the Simulacra Project

## Vowel Harmony

If agents backprogogate with topographic similarity in mind and the basic signals (phonemes) are endowed with a similarity they may end up with systems with vowel harmony and alternation of consonants to capture sets normal subgroups with greater similarity.

if these regular configuration also lead to better channel coding the benefits should persist.

## VO and OV via symmetry breaking etc

If we use Huffman coding like process to organize the order of the morphological and syntactical elements (effectively making the fixing the on average most surprising partial signals before the next on average most surprising ones) we should have emergent languages that are rather similar and fairly easy to learn

Like Turkish and Japanese. However there is at the start the question of how to apply aggregations. If action is first we get V O languages if it is second we get OV languages. I think that V carries more entropy in Predation and Resource gathering games so that VO should be more prevalent. However once this decision is made most practical algorithms will not be able to reverse it.

Arora, Sanjeev, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. 2018. *Linear Algebraic Structure of Word Senses, with Applications to Polysemy*. <https://arxiv.org/abs/1601.03764>.

Bloom, L. 2013. *One Word at a Time: The Use of Single Word Utterances Before Syntax*. Janua Linguarum. Series Minor. De Gruyter. <https://books.google.co.il/books?id=4bTnBQAAQBAJ>.

Bod, R., J. Hay, and S. Jannedy. 2003. *Probabilistic Linguistics*. A Bradford Book. MIT Press. <https://books.google.co.il/books?id=Ac5bgYVk8kwC>.

Dryer, Matthew S., and Martin Haspelmath, eds. 2013. *WALS Online (V2020.4)*. Data set. Zenodo. <https://doi.org/10.5281/zenodo.13950591>.

Fenson, L. 2000. *Variability in Early Communicative Development*. Monographs of the Society for Research in Child Development. Wiley. <https://books.google.co.il/books?id=9N0cZ24IzIUC>.

Foerster, Jakob, Ioannis Alexandros Assael, Nando de Freitas, and Shimon Whiteson. 2016. “Learning to Communicate with Deep Multi-Agent Reinforcement Learning.” In *Advances in Neural Information Processing Systems*, edited by D. Lee, M. Sugiyama, U. Luxburg, I. Guyon, and R. Garnett, vol. 29. Curran Associates, Inc. <https://proceedings.neurips.cc/paper_files/paper/2016/file/c7635bfd99248a2cdef8249ef7bfbef4-Paper.pdf>.

Gold, E Mark. 1967. “Language Identification in the Limit.” *Information and Control* 10 (5): 447–74.

Goldsmith, John. 2001. “Unsupervised Learning of the Morphology of a Natural Language.” *Computational Linguistics* (Cambridge, MA) 27 (2): 153–98. <https://doi.org/10.1162/089120101750300490>.

Hart, B., and T. R. Risley. 1995. *Meaningful Differences in the Everyday Experience of Young American Children*. P.H. Brookes. <https://books.google.co.il/books?id=I2pHAAAAMAAJ>.

Hoff, Erika. 2009. *Language Development*. 5th ed. Cengage Learning.

MacKay, David J. C. 2003. *Information Theory, Inference, and Learning Algorithms*. Copyright Cambridge University Press.

Nowak, Martin A., Natalia L. Komarova, and Partha Niyogi. 2002. “Computational and Evolutionary Aspects of Language.” *Nature* 417 (6889): 611–17. <https://api.semanticscholar.org/CorpusID:4323969>.

Nowak, Martin A, and David C Krakauer. 1999. “The Evolution of Language.” *Proceedings of the National Academy of Sciences* 96 (14): 8028–33.

Skyrms, Brian. 2010. “14512 Complex Signals and Compositionality.” In *Signals: Evolution, Learning, and Information*. Oxford University Press. <https://doi.org/10.1093/acprof:oso/9780199580828.003.0013>.

Today, Robotics. 2021. “Learning to Communicate in Multi-Agent Systems.” In *Https://Www.youtube.com/@RoboticsTodaySeminar*. <https://www.youtube.com/watch?v=J9Olp6YqQhw&ab_channel=RoboticsToday>.

## Citation

BibTeX citation:

``` quarto-appendix-bibtex
@online{bochman2025,
  author = {Bochman, Oren},
  title = {Emergent {Languages}},
  date = {2025-01-14},
  url = {https://orenbochman.github.io/posts/2025/2025-01-14-Desiderata-for-Emergent-Languages/},
  langid = {en}
}
```

For attribution, please cite this work as:

Bochman, Oren. 2025. “Emergent Languages.” January 14. <https://orenbochman.github.io/posts/2025/2025-01-14-Desiderata-for-Emergent-Languages/>.

[^1]: 4.7 years at 1 step per minute

[^2]: perhaps not so easy as N increases to millions and there are a a small chance of errors per symbol

[^3]: for lewis signaling games where agents learn a lexicon this is is all an agent needs to learn the signaling system.

[^4]: As it overgenerates, it might be necessary to store a bit-list of which lexicon items can be used with which templates. though I don’t actually recall this being the case as the might be an infinite number.

[^5]: I have already written an article on some different ways that signaling systems can be arise.

[^6]: a word is characterized by the company it keeps

[^7]: explain!?

[^8]: multiple learners has similar logic as a multiple hypothesis testing problem, for each learner postulating different signaling system with each failure or success in a Lewis game. More so when learners get to observe each other’s actions and rewards.
