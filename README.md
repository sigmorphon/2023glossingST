# SIGMORPHON 2023 Shared Task on Interlinear Glossing 

## Overview

Interlinear glossed text is a major annotated datatype produced in the course of linguistic fieldwork. For many low-resource languages, this is the only form of annotated data that is available for NLP work. Creation of glossed text is, however, a laborious endeavour and this shared task investigates methods to (fully or partially) automate the process. 

Participants build systems which generate morpheme-level grammatical descriptions of input sentences following the [Leipzig glossing conventions](https://www.eva.mpg.de/lingua/pdf/Glossing-Rules.pdf). The input to the glossing system consists of (1) a sentence in the target language and (2) a translation of the target sentence into a language of wider communication, often English (plus some additional information which is discussed below). The output is an interlinear gloss:

<table>
<tr><td>1. Input 1 (source):</td> <td>Ii</td> <td>k̲'ap</td> <td>g̲aniwila</td> <td>yukwhl</td> <td>surveyors</td></tr>
<tr><td>2. Ouput (gloss):</td><td> CCNJ </td><td>VER </td><td>continually-MANR</td><td> do-CN</td><td> surveyors</td></tr>
<tr><td>3. Input2 (translation):</td><td colspan="5"> ‘But the surveyors continued.’</td></tr>
</table>
Figure 1: A glossed sentence in the Gitksan language 

As demonstrated in Figure 1, bound morphemes like *-hl* are glossed using morphological tags like CN (common noun connective) and word stems like *yukw* are glossed with a translation (here, English ‘do’). 

Participants are encouraged to draw inspiration from existing glossing systems: Barriga et al. (2021), Macmillan-Major (2020), Moeller and Hulden (2018), Palmer et al. (2009), Samardžić et al. (2015) and Zhao et al. (2020)

## Sign up

Sign up for the shared task by filling in the [registration form](https://docs.google.com/forms/d/e/1FAIpQLSe3K7lFK2FNJooeMlb6ffhWQVZzy257zYNkbOEoBXePFkp6uQ/viewform).

## Tracks

There are two tracks in the shared task. In the closed track (`track1`), systems are trained solely on input sentences and glosses. In the open track (`track2`), systems may additionally make use of morphological segmentations during training time. In the open track, participants may additionally use any data and resources (including dictionaries and pretrained language models). The only exception is additional interlinear glossed data. For the open track, we also provide some extra information like POS tags for a subset of the languages. 

If you are at all unsure whether some data is allowed, **we recommend that you contact the organizers**. 

## Data

### Source

Some of our datasets were collected and annotated by the shared task organizers. Others come from published works. All of the data has been carefully manually annotated by competent linguists. We will reveal the source of all datasets after the evaluation period has been concluded.

The following languages are released as development languages. Additional surprise language data will be released later. for some of the languages, we additionally release morphological segmentations, POS tags and translations which are available for training/testing systems depending on the track (closed vs. open track).

| Language       | Train sents | Dev sents | Test sents | Morph. Segmentations? | POS tags? | Translations? |
|----------------|-------------|-----------|------------|-----------------------|-----------|---------------|
| Arapaho (arp)  |  39,501     |  4,938    |    TBA     |                       |           | X  (eng)      |
| Gitksan (git)  |  31         |  42       |    TBA     |         X             |           | X  (eng)      |
| Lezgi   (lez)  |  701        |  88       |    TBA     |         X             |           | X  (eng)      |
| Nyangbo (nyb)  |  2,100      |  263      |    TBA     |         X             |           |               |
| Tsez    (bbo)  |  3,558      |  445      |    TBA     |         X             |           | X  (eng)      |
| Uspanteko (usp)|  9,774      |  232      |    TBA     |         X             |    X      | X  (spa)      |

Note that translations are not provided for Nyangbo and the translations for Uspanteko are in Spanish, not English.

### Format

#### Track1 – closed track

Data sets for training (e.g. `data/Tsez/ddo-train-track1-uncovered`) and evaluation (e.g. `data/Tsez/ddo-dev-track1-uncovered`) follow this format:

```
\t Retinäy debex mi yižo, retinäy q’ˤuyzix yegir.
\g IV-want-CND.CVB you-AD.ESS you II-lead-IMPR IV-want-CND.CVB other-ATTR.OBL-AD.ESS II-send
\l If you want, marry her yourself, or if you want, send her to someone else.

\t Esnazał xizaz ixiw raład boqno.
\g sister-PL-CONT.ESS behind big sea III-become-PST.UNW
\l And a big sea formed behind the sisters.
```

Individual glossed sentences are separated by empty lines.

Each line identifies a different type of information:

* `\t` orthographic representation
* `\g` gold standard gloss
* `\l` English (or Spanish) translation  

We additionally provide system input files (e.g. `data/ddo-dev-track1-covered`), where the gold standard gloss is missing: 

```
\t Retinäy debex mi yižo, retinäy q’ˤuyzix yegir.
\g 
\l If you want, marry her yourself, or if you want, send her to someone else.

\t Esnazał xizaz ixiw raład boqno.
\g 
\l And a big sea formed behind the sisters.
```

#### Track 2 – open track

Data sets for training (e.g. `data/Tsez/ddo-train-track2-uncovered`) and evaluation (e.g. `data/Tsez/ddo-dev-track2-uncovered`) follow this format:

```
\t Retinäy debex mi yižo, retinäy q’ˤuyzix yegir.
\m r-eti-näy mi-x mi y-iži-o r-eti-näy q’ˤuya-zo-x y-egir
\g IV-want-CND.CVB you-AD.ESS you II-lead-IMPR IV-want-CND.CVB other-ATTR.OBL-AD.ESS II-send
\l If you want, marry her yourself, or if you want, send her to someone else.

\t Esnazał xizaz ixiw raład boqno.
\m esyu-bi-ł xizaz ixiw raład b-oq-n
\g sister-PL-CONT.ESS behind big sea III-become-PST.UNW
\l And a big sea formed behind the sisters.
```

Each line identifies a different type of information:

* `\t` orthographic representation
* `\m` morphological segmentation
* `\g` gold standard gloss
* `\l` English (or Spanish) translation  

For a subset of the languages, we will also provide an additional *POS annotation tier* (`\p`) for training purposes:

```
\t o sey xtok rixoqiil
\m o' sea x-tok r-ixóqiil
\p CONJ ADV COM-VT E3S-S
\g o sea COM-buscar E3S-esposa
\l O sea busca esposa.
```
 
We additionally provide system input files (e.g. `data/ddo-dev-track2-covered`), where the gold standard gloss is missing. These files contain morphological segmentations but no POS annotations:

```
\t Retinäy debex mi yižo, retinäy q’ˤuyzix yegir.
\m r-eti-näy mi-x mi y-iži-o r-eti-näy q’ˤuya-zo-x y-egir
\g 
\l If you want, marry her yourself, or if you want, send her to someone else.

\t Esnazał xizaz ixiw raład boqno.
\m esyu-bi-ł xizaz ixiw raład b-oq-n
\g 
\l And a big sea formed behind the sisters.
```

## Evaluation

The main evaluation metric for the competition is token accuracy. Systems are evaluated w.r.t. generation of fully glossed tokens (*chiens -> dog-PL*). We will also separately evaluate glossing accuracy on bound morphemes like *PL* and free morphemes, i.e. stems, like *dog*.

## Submission

At the end of April, we will release the test input data in the following format (for track 1 in the example):

```
\t ʕAt’idä nesiq kinaw raqru łinałäy esin.
\g 
\l Atid told about everything that had happened to him.

\t Ražbadinez idu barun, xexbin yołƛin, žawab teƛno ečruni žek’a.
\g 
\l "His wife and children live at Razhbadin's home", answered the old man.
```

Participants use their glossing system to predict glosses for the tokens in the test data and submit their predictions to the shared task organizers (EMAIL address) in the following format:

```
\t ʕAt’idä nesiq kinaw raqru łinałäy esin.
\g Atid-ERG DEM1.ISG.OBL-POSS.ESS entire IV-happen-PST.PRT what.OBL-CONT.ABL tell-PST.UNW
\l Atid told about everything that had happened to him.

\t Ražbadinez idu barun, xexbin yołƛin, žawab teƛno ečruni žek’a.
\g Razhbadin-GEN2 home wife-and children-and be-QUOT answer give-PST.UNW old-DEF man-ERG
\l "His wife and children live at Razhbadin's home", answered the old man.
```

## Baseline

In early March, we will release baseline systems and results for both tracks. For the closed track (track 1), we will provide a transformer-based neural baseline system. For the open track (track 2), we will provide CRF-based and neural transformer baseline systems.  

## Organizers

* Michael Ginn (University of Colorado)
* Mans Hulden (University of Colorado)
* Sarah Moeller (University of Florida)
* Garrett Nicolai (University of British Columbia)
* Alexis Palmer (University of Colorado)
* Miikka Silfverberg (University of British Columbia)
* Anna Stacey (University of British Columbia)

## Contact

You can email the shared task organizers: `sigmorphonglossingst2023@gmail.com`

Please also subscribe to the shared task newsgroup: https://groups.google.com/g/sigmorphonglossingst2023

## Timeline 

* Feb 13: Release of training and development data for development languages
* March 6: Release of official evaluation script, baseline systems and baseline results
* April 1: Release of surprise language training and development data
* April 24: Release of test data for all languages
* April 24-26: Contestants run their systems on the test data
* April 27: Test predictions should be submitted to organizers
* May 1: Results are announced
* May 15: System description paper submission deadline
* May 15-25: Review
* May 25: Notification of paper acceptance
* May 30: Camera ready deadline for system description papers

## Licensing

All baseline and evaluation code is released under the Apache 2.0 license. Each dataset is released under a separate license which can be found in the data/LAN directory.

## References

Baldridge, J., & Palmer, A. (2009, August). [How well does active learning actually work? Time-based evaluation of cost-reduction strategies for language documentation](https://aclanthology.org/D09-1031/). In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing (pp. 296-305).

Barriga, D., Mijangos, V., & Gutierrez-Vasques, X. (2021). [Automatic Interlinear Glossing for Otomi language](https://aclanthology.org/2021.americasnlp-1.5/). NAACL-HLT 2021, 34.

Edwards, B., Larochelle, M., Mitchell, S., Van Eijk, J., Davis, H., Lyon, J. and Whitley, R.S. (2017). [Sqwéqwel’s Nelh Skelkekla7lhk{\'a}lha Tales of Our Elders](https://lingpapers.sites.olt.ubc.ca/pnwll-volumes/sqweqwels-nelh-skelkekla7lhkalha-tales-of-our-elders/). University of British Columbia Occasional Papers in Linguistics

Lewis, W. D., & Xia, F. (2010). [Developing ODIN: A multilingual repository of annotated language data for hundreds of the world's languages](https://academic.oup.com/dsh/article-abstract/25/3/303/971097?redirectedFrom=fulltext). Literary and Linguistic Computing, 25(3), 303-319.

McMillan-Major, A. (2020). [Automating gloss generation in interlinear glossed text](https://aclanthology.org/2020.scil-1.42/). Proceedings of the Society for Computation in Linguistics, 3(1), 338-349.

Moeller, S., & Hulden, M. (2018, August). [Automatic glossing in a low-resource setting for language documentation](https://aclanthology.org/W18-4809/). In Proceedings of the Workshop on Computational Modeling of Polysynthetic Languages (pp. 84-93).

Palmer, A., Moon, T., & Baldridge, J. (2009, June). [Evaluating automation strategies in language documentation](https://aclanthology.org/W09-1905/). In Proceedings of the NAACL HLT 2009 Workshop on Active Learning for Natural Language Processing (pp. 36-44).

Zhao, X., Ozaki, S., Anastasopoulos, A., Neubig, G., & Levin, L. (2020, December). [Automatic interlinear glossing for under-resourced languages leveraging translations](https://aclanthology.org/2020.coling-main.471/). In Proceedings of the 28th International Conference on Computational Linguistics (pp. 5397-5408).

Samardžić, T., Schikowski, R., & Stoll, S. (2015). [Automatic interlinear glossing as two-level sequence classification](https://aclanthology.org/W15-3710/).
