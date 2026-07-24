# Fine-Tuning an LLM into a FAR Institutional-Expert Model: Deep Dive on Methods and Failure Modes

## Overview

Turning a general-purpose LLM into a model that deeply understands one institution — the Forces Armées Royales' terminology, organizational architecture, administrative processes, and internal/civil legal framework — while retaining general reasoning ability is not a single fine-tuning run. It is a multi-stage pipeline (continued pretraining → supervised fine-tuning → preference alignment) where every stage introduces its own failure mode, and the failure modes compound. This report is grounded in a specific target: **Qwen3-VL-30B-A3B-Instruct** (fine-tuning target) running on **2× NVIDIA H200 141GB (PCIe, PP=2)** with the **inference-cluster-stack** (vLLM + Qdrant + Docker Compose, not Ollama). It goes deeper into the mechanics of why forgetting happens, what actually works to mitigate it, why the training data itself is harder to build than expected for an institutional/legal corpus specifically, why hallucination risk can *increase* rather than decrease from fine-tuning, and why measuring success is far less settled than a standard benchmark suite suggests. The underlying fine-tuning mechanics are domain-agnostic and drawn from the same body of evidence as any deep model-adaptation project; every case study, data-curation problem, and evaluation gap has been re-grounded here in the FAR/Moroccan-military-legal-HR domain.

<!-- Model mismatch warning removed — stack will be updated to 30B -->

## Target Model Architecture: Qwen3-VL-30B-A3B

This document is grounded in a specific model: **Qwen3-VL-30B-A3B-Instruct** (QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ for inference; Qwen/Qwen3-VL-30B-A3B-Instruct BF16 for training). Key architectural facts from the official config.json:

| Property | Value | Implication |
|---|---|---|
| Total params | ~31B (30.3B active) | Fits on one H200 in AWQ (~17 GB) or BF16 (~61 GB) |
| Architecture | MoE, 128 experts, 8 active | Requires DeepSpeed ZeRO-2 (not ZeRO-3) for LoRA training |
| Layers | 48 | PP=2 splits 24 layers per GPU |
| KV heads | 4 | Small KV head count means smaller KV cache per token (4 × 128 × 1 byte FP8 = 512 bytes/token) |
| Head dim | 128 | Standard size, FlashAttention-2 compatible |
| Hidden size | 2048 | Relatively compact for a 30B model |
| Max position embeddings | 262,144 (native 256K) | No YaRN needed for 64K context — native support |
| Vision encoder depth | 27 | Frozen during training (`tune_mm_vision=False`) |
| Projector | MLP cross-attention | Fine-tuned (`tune_mm_mlp=True`) |
| Tokenizer vocab | ~152K tokens | Large vocab makes token additions costly — only add tokens with >100 corpus occurrences |
| Dtype for training | BF16 | Base checkpoint is BF16; AWQ is inference-only |

**Variant choice: Instruct over Thinking.** The Thinking variant produces 2-3× more output tokens per query (chain-of-thought deliberation before answer). At 64K context and 250 concurrent users, this compounds KV cache pressure by 2-3×, pushing more users into CPU swap. Instruct gives direct, citation-grounded answers with ~1/3 the token overhead, which is the correct tradeoff for a high-concurrency institutional Q&A system.

---
## FAR Domain Knowledge Base: Vocabulary, Structure, and Institutional Terminology

The model must understand FAR's institutional language across three languages (French, Darija, MSA Arabic) and dozens of specialized sub-domains. This section documents the key vocabulary, organizational structures, legal instruments, and typical query patterns the model will encounter — sourced from official FAR publications (revue.far.ma), the Bulletin Officiel, UN peacekeeping documentation, and Moroccan government legal databases.

### Organizational hierarchy and command structure

The FAR is organized under the King as Supreme Commander and Chief of Staff, with a civilian Minister-Delegate for National Defense Administration and a military Inspector General. Key entities:

| Entity | French name | Arabic name | Function |
|---|---|---|---|
| FAR High Command | Commandement général des FAR | القيادة العامة للقوات المسلحة الملكية | Overall command and strategy |
| Inspector General | Inspecteur Général des FAR | المفتش العام للقوات المسلحة الملكية | Senior military commander (currently Lt Gen Mohammed Berrid) |
| National Defense Administration | Administration de la Défense Nationale | إدارة الدفاع الوطني | Civilian oversight (Minister Abdellatif Loudiyi) |
| 2nd Bureau | Deuxième Bureau | المكتب الثاني | Military intelligence |
| 3rd Bureau | Troisième Bureau | المكتب الثالث | Joint operations coordination |
| 4th Bureau | Quatrième Bureau | المكتب الرابع | Equipment procurement and logistics |
| 5th Bureau | Cinquième Bureau | المكتب الخامس | Military security |
| Directorate of General Studies | Direction Générale des Études et de la Documentation | المديرية العامة للدراسات والوثائق | Strategic intelligence (civilian-led) |

**Service branches:**

| Branch | French | Arabic | Personnel |
|---|---|---|---|
| Royal Army | Armée Royale | القوات البرية الملكية | ~175,000 (incl. conscripts) |
| Royal Air Force | Forces Royales Air | القوات الجوية الملكية | ~13,000 |
| Royal Navy | Marine Royale | القوات البحرية الملكية | ~12,000 |
| Royal Gendarmerie | Gendarmerie Royale | الدرك الملكي | ~23,000 |
| Royal Guard | Garde Royale | الحرس الملكي | ~3,500 |
| Auxiliary Forces | Forces Auxiliaires | القوات المساعدة | ~30,000 |

**Key organizational acronyms the model must resolve:**

| Acronym | Full name | Role |
|---|---|---|
| DAG | Direction des Affaires Générales | General affairs — civil/military personnel, matériel, budget |
| DGM | Direction Générale... Discipline Générale Militaire | General Military Discipline (the code itself, Dahir 1-74-383) |
| DGSN | Direction Générale de la Sûreté Nationale | National Security (civilian police, often cross-referenced) |
| TSD | Tribunal Spécial... Travaux... | Special Court... (varies by context) |
| BSN | Bureau Spécial... | Special Bureau |
| BSP | Bureau de Sécurité... | Security Bureau |
| DHR | Direction des Ressources Humaines | Human Resources Directorate |
| DAJ | Direction des Affaires Juridiques | Legal Affairs Directorate |
| DBF | Direction du Budget et des Finances | Budget and Finance Directorate |
| CMR | Caisse Marocaine des Retraites | Moroccan Pension Fund (manages military pensions) |
| HMIMV | Hôpital Militaire d'Instruction Mohammed V | Mohammed V Military Teaching Hospital |
| EMG | État-Major Général | General Staff |
| GMR | Groupement... (various unit types) | Grouping/Battalion-level unit |

### Rank structure (French / Arabic / NATO code)

The rank system follows the French model with a unique **Colonel-Major** rank (created mid-1970s post-coup attempts, between Colonel and Général de Brigade). The FAR conducts all military business in French; Arabic is used for formal correspondence with the monarchy.

**Commissioned officers (officiers):**

| NATO | French | Arabic | Notes |
|---|---|---|---|
| OF-9 | Général d'Armée | فريق أول | Army General |
| OF-8 | Général de Corps d'Armée | فريق | Corps General (4 stars) |
| OF-7 | Général de Division | لواء | Division General (3 stars) |
| OF-6 | Général de Brigade | عميد | Brigade General (2 stars) |
| OF-5 (senior) | Colonel Major | عقيد قائد | **Unique to FAR** — commands brigades of 3K-5K troops |
| OF-5 | Colonel | عقيد | Colonel |
| OF-4 | Lieutenant Colonel | مقدم | Lieutenant Colonel |
| OF-3 | Commandant | رائد | Major (called Commandant in French tradition) |
| OF-2 | Capitaine | نقيب | Captain |
| OF-1 | Lieutenant | ملازم | Lieutenant |
| OF-1 (junior) | Sous Lieutenant | ملازم متمرن | Second Lieutenant |

**Non-commissioned officers and enlisted (sous-officiers et militaires du rang):**

| French | Arabic |
|---|---|
| Adjudant Chef | مساعد أول |
| Adjudant | مساعد |
| Sergent Chef | رقيب أول |
| Sergent | رقيب |
| Caporal Chef | عريف أول |
| Caporal | عريف |
| Soldat de 1re classe | جندي درجة أولى |
| Soldat de 2e classe | جندي درجة ثانية |

**In Darija** (street/operational context), ranks are often adapted phonetically from French or use Arabic terms with Darija pronunciation: `l'jeneral`, `l'colonel`, `l'capitaine`, `s'rgay` (sergent), `l'commandant`.

### Legal framework: key instruments and cross-reference patterns

The FAR legal framework is a hierarchy of dahirs (royal decrees), lois (parliamentary laws), décrets (government decrees), and internal directives. Every legal text has a standard citation format: **Dahir n° X-XX-XXX du date** — this pattern MUST be understood by the tokenizer and model.

| Instrument | Number | Subject | Cross-references |
|---|---|---|---|
| Discipline Générale Militaire | Dahir 1-74-383 (5 Aug 1974) | General discipline in FAR | Cited by Code de Justice Militaire, Loi 01-12, Loi 44-18 |
| Code de Justice Militaire | Dahir 1-56-270 (10 Nov 1956) | Military courts, procedure, penalties | References DGM articles 18-21 (investigation powers), 96 (disciplinary council), 86 (contract termination) |
| Garanties Fondamentales | Loi 01-12 (projet) | Fundamental rights of FAR personnel | Cross-refs Constitution articles 6, 21-23, 59, 117-118, 155; DGM art. 18-21; Code Pénal art. 124 |
| Service Militaire | Loi 44-18 (25 Jan 2019) | Reinstated compulsory military service, 12 months, age 19-25 | Subjects conscripts to DGM + Code Justice Militaire + Loi 01-12 simultaneously |
| Statut Particulier des Officiers | Dahir 1-12-50 (2012) | Specific status of FAR officers | Links to civil service statute, promotion conditions, retirement |
| Pensions Militaires | Loi 013-71 (30 Dec 1971) | Military pension regime, CMR-managed | Rates, eligibility, reversion rights (50% to spouse) |
| Code Pénal | Dahir 1-59-413 (26 Nov 1962) | Criminal code — cited for sexual exploitation (arts. 484-488), general criminal acts by military personnel |

**Cross-reference patterns the model must handle:**

The model will encounter multi-instrument questions like: *"Un conscrit sous Loi 44-18 commet une infraction pendant le service. Quel instrument régit sa situation? La DGM, le Code de Justice Militaire, ou les deux? Et la Loi 01-12 s'applique-t-elle pendant un arrêt de forteresse?"*

The answer requires traversing:
- Loi 44-18 art. 1-4 (jurisdiction: conscripts subject to military law)
- DGM 1-74-383 art. 86-96 (disciplinary sanctions, procedure)
- Code de Justice Militaire 1-56-270 art. 3-4 (personal jurisdiction of military tribunals)
- Loi 01-12 art. 7 (criminal responsibility during operations — does not apply during disciplinary detention)

**Common legal citation formats in French:**

| Pattern | Example |
|---|---|
| Full dahir citation | `Dahir n° 1-74-383 du 15 rejeb 1394 (5 août 1974)` |
| Article reference | `Article 86 du Dahir 1-74-383` |
| Multi-article chain | `Articles 18, 19, 20 et 21 du Dahir n° 1-74-383` |
| Cross-text reference | `sous réserve des dispositions de l'article 12 du Dahir n° 1-12-50` |
| BO citation | `Bulletin Officiel, n° 3240 bis, 9 décembre 1974, p. 1685-1701` |
| Application decree | `Décret n° 2-19-46 du 13 joumada II 1440 (19 février 2019) fixant les modalités d'application de la loi n° 44-18` |

### HR and personnel administration terminology

This is the highest-query-frequency domain. Key terms in French (the primary administrative language), with Darija equivalents where they exist:

| French term | Darija equivalent | English |
|---|---|---|
| Engagement / Rengagement | l'engagement / r'engagement | Enlistment / Re-enlistment |
| Contrat à durée déterminée | contrat m'hodad | Fixed-term contract |
| Résiliation de plein droit | résiliation b'l'haq | Automatic termination (DGM art. 86) |
| Faute contre l'honneur | ghalta d'd l'chraf | Offence against honour |
| Inconduite habituelle | soulouk mouchine | Habitual misconduct |
| Conseil de discipline | conseil d'discipline | Disciplinary board |
| Arrêts de forteresse | l'arrêt d'forteresse | Confinement to barracks (military punishment) |
| Radiation des cadres | radiation | Removal from rolls |
| Mise à la retraite d'office | retraite forcée | Compulsory retirement |
| Limite d'âge | had d'sinn | Age limit (varies by rank: 41 for Lt, 61 for Général) |
| Congé de maladie | congé m'réd | Sick leave |
| Congé de convalescence | congé d'convalescence | Recovery leave |
| Permission exceptionnelle | permission istithnaïya | Exceptional leave |
| Avancement au choix | taraqqi b'l'ikhtiyar | Merit-based promotion |
| Avancement à l'ancienneté | taraqqi b'l'aqdamiya | Seniority-based promotion |
| Notation | taqyim | Performance evaluation |
| Tableau d'avancement | lawhat t'taraqqi | Promotion list |

### Pension and compensation terminology

Military pensions are managed by the CMR (Caisse Marocaine des Retraites) under Loi 013-71. Key terms:

| French term | Meaning | Key detail |
|---|---|---|
| Pension militaire | Military pension | Managed by CMR, distinct from civil service pension |
| Solde spéciale progressive | Progressive special pay | Base for pension calculation |
| Taux de pension | Pension rate | 90% for caporaux, 80% for soldats (of caporal-chef baseline) |
| Pension de réversion | Survivor's pension | 50% to surviving spouse, 25% per orphan child |
| Annuités liquidables | Pensionable years | Years of service counting toward pension |
| Dernier traitement de base | Final base salary | Used for CMR calculation formula: 2% × years × base salary |
| Plafond de pension | Pension cap | 75% of base salary (post-2016 reform) |
| Décote | Reduction | -1.25% per missing quarter |
| Surcote | Bonus | +1.25% per additional quarter |
| Allocation de fin de carrière | End-of-career allowance | Lump sum if below minimum vesting period |
| Rachat d'années | Year buyback | Purchase of military service or higher education years |
| CMR | Caisse Marocaine des Retraites | Manages both civil and military pension regimes |
| Atakmili | Complementary CMR regime (تكميلي) | Voluntary supplementary pension since 2017 |

**Pension formula (post-2016 reform):** `Pension brute = 2% × Salaire de référence × Nombre d'années de service`, capped at 75% of base salary (achieved at 50 years of service). Pre-2016: 2.5% per year, 100% cap at 40 years.

### Disciplinary and sanctions terminology

| French term | Meaning | Legal basis |
|---|---|---|
| Faute disciplinaire | Disciplinary offence | DGM art. 1-17 |
| Sanction disciplinaire | Disciplinary sanction | DGM art. 18-96 |
| Avertissement | Warning | Lowest sanction |
| Blâme | Reprimand | Written censure |
| Exclusion temporaire | Temporary suspension | With or without pay |
| Rétrogradation | Demotion | Rank reduction |
| Exclusion définitive | Permanent dismissal | From the institution |
| Arrêts de forteresse | Confinement | 1-60 days typically |
| Conseil d'enquête | Investigation board | Art. 96 DGM |
| Non-lieu | No case to answer | Dismissal of proceedings |
| Classement sans suite | File closed | No further action |
| Pourvoi en grâce | Appeal for clemency | To the King |

### Operational and equipment terminology

| French | Arabic / Darija | Meaning |
|---|---|---|
| Engagement opérationnel | iltizam 'amaliyati | Operational deployment |
| Opération de maintien de la paix | 'amaliyat salam | Peacekeeping (MONUC, ONUCI, EUFOR, KFOR, MINUSCA) |
| Exercice conjoint | tamrin mushtarak | Joint exercise (e.g., African Lion, "أسد أفريقيا") |
| Zone Est | al-mintaqa al-sharqiya | Eastern Military Zone |
| Zone Sud | al-mintaqa al-janubiya | Southern Military Zone (Western Sahara) |
| Défense aérienne | difa' jawwi | Air defence (Barak MX, Skyguard, MICA) |
| Renseignement militaire | istikhbarat 'askariya | Military intelligence (2e Bureau) |
| Contre-insurrection | mukafahat al-tamarrud | Counter-insurgency (Sahara expertise) |
| Guerre en montagne | harb jabaliya | Mountain warfare |
| Guerre du désert | harb sahra'iya | Desert warfare |

### Darija military and administrative vocabulary

Darija is the language of daily operational communication, oral orders, informal HR queries, and soldier-to-administrator interaction. It has no standardized orthography — the model must handle Latin-script Darija (Arabizi), Arabic-script Darija, and mixed French-Darija.

**Common Darija administrative terms:**

| Darija (Latin) | Darija (Arabic) | Meaning | Source language |
|---|---|---|---|
| l'khdma | الخدمة | Work/service | Arabic |
| l'prime | البريمة | Bonus/allowance | French (prime) |
| l'congé | الكونجي | Leave/holiday | French (congé) |
| l'avancement | لافونسمون | Promotion | French (avancement) |
| l'flous | الفلوس | Money/pay | Darija/Arabic |
| l'papiers | الپاپيي | Documents/papers | French (papiers) |
| s'ghar | السڭار | Cigarettes (informal currency) | French (cigare) |
| l'permis | الپيرمي | Permission/permit | French (permis) |
| l'm'khazni | المخزني | Government official/soldier | Arabic (related to Makhzen) |
| l'qayd | القايد | Commander/chief | Arabic |
| s'rgay | السارڭي | Sergeant | French (sergent) |
| l'mou9awem | المقاوم | Resistant/fighter | Arabic |
| ch'hal f'l'grade? | شحال ف لڭراد؟ | What's your rank? | Mixed French-Darija |
| ch'hal f'l'prime? | شحال ف لپريم؟ | How much is the bonus? | Mixed French-Darija |
| 3tini l'congé | عطيني لكونجي | Give me leave | Mixed Arabic-French |
| baghi n'khdem f... | باغي نخدم ف... | I want to work in... | Darija |
| m3ak l'papiers? | معاك لپاپيي؟ | Do you have the documents? | Mixed |
| labas 3la l'khdma? | لاباس على الخدمة؟ | How's the work? | Darija |

**Code-switching patterns the model must handle:**

The most realistic query mode mixes French nouns into Darija grammatical frames:
- `"baghi n'taleb l'congé d'maladie"` — "I want to request sick leave" (Darija verb frame + French noun phrase)
- `"ch'hal mn année f'l'CMR?"` — "How many years in the CMR?" (Darija quantifier + French acronym)
- `"l'capitaine ila baghi y'khdem promotion"` — "The captain wants to work on his promotion" (mixed)
- `"ana sergent, 3ndi 15 ans f'l'khdma, ch'hal ghadi t'koun l'pension?"` — "I'm a sergeant, I have 15 years of service, how much will the pension be?" (fully mixed)

### Common query types and expected answer format

Based on administrative HR patterns in comparable military institutions, the model will most frequently encounter:

| Query type | Example | Expected answer format |
|---|---|---|
| Pension calculation | "I'm a Captain with 22 years of service, retiring at 55. What CMR pension?" | Formula + estimated range + legal basis |
| Leave entitlement | "How many days of sick leave can a sergeant get per year?" | Legal limit + conditions + applicable dahir |
| Promotion criteria | "What conditions for merit-based promotion from Captain to Commandant?" | Service years + evaluation + board procedure |
| Disciplinary procedure | "What happens if a conscript refuses orders?" | DGM articles + possible sanctions + appeal |
| Legal cross-reference | "Loi 44-18 says I'm subject to the DGM. Does Article 13 of Loi 01-12 apply during arrêts?" | Multi-hop reasoning across 3-4 instruments |
| Document request | "I need a copy of my service record for CMR pension application" | Procedure + office + timeline |
| Conscription info | "I'm 22, MRE living in France. Can I volunteer for military service?" | Eligibility + Tajnid platform + procedure |
| Benefit query | "What benefits do retired soldiers get?" | Healthcare + pension + housing + education |
| Language request | "تكلم معايا بالدارجة" or "Parle-moi en darija" | Switch to specified language |
| Translation | "What does 'radiation des cadres' mean in Darija?" | Term + translation + explanation |

### Domain-specific document types

The model's training data and RAG corpus will include these document types, each with distinct formatting and language patterns:

| Document type | Language(s) | Format characteristics | Example |
|---|---|---|---|
| Dahir (Royal Decree) | French, Arabic | `Dahir n° X-XX-XXX du date`, BO citation | Dahir 1-74-383 |
| Loi (Law) | French, Arabic | `Loi n° XX-XX`, parliamentary vote date | Loi 44-18 |
| Décret d'application | French | `Décret n° X-XX-XX`, ministerial signature | Décret 2-19-46 |
| Bulletin Officiel | French, Arabic | BO number, date, page range | BO 3240 bis, 1685-1701 |
| État de service | French | Table format, rank/date/unit columns | Service record |
| Note de service | French | Internal memo format | Commander's directive |
| Décision individuelle | French | Named individual, effective date | Promotion/transfer |
| Convocation | French, Arabic | Template, date/time/location | Conscription summons |
| Règlement intérieur | French | Section/article structure | Unit standing orders |
| Compte rendu | French | Narrative, date/signature block | Incident report |
| Procès-verbal | French | Formal minutes, witness signatures | Disciplinary hearing |
| Certificat médical | French | Doctor's stamp, diagnosis, duration | Medical leave |
| Formulaire CMR | French | Structured fields, tax references | Pension application |
| Revue des FAR | French | Magazine format, articles, photos | revue.far.ma |

---
## 1. Why Catastrophic Forgetting Happens (Not Just That It Happens)

Catastrophic forgetting is usually described as "the model overwrites old knowledge," but recent mechanistic work identifies three distinct, measurable causes — all directly relevant to a model that needs to retain both general reasoning and precise institutional recall.

**Representation drift.** Continual training measurably destroys a model's syntactic and semantic representations of the same inputs it saw before training, and this destruction directly predicts worse downstream performance later ([Investigating Forgetting in Pre-Trained Representations](https://arxiv.org/abs/2305.05968)). Knowledge-circuit analysis shows new-knowledge acquisition depends on how related it is to existing knowledge, and that the circuits responsible evolve from deep layers to shallow layers as training progresses ([Knowledge Circuits Perspective](https://arxiv.org/abs/2502.11196)). For a FAR model, this means teaching genuinely new institutional facts (a specific article number in the Discipline Générale Militaire, a DAG division name) is mechanistically harder for the model to integrate cleanly than teaching it to phrase existing general knowledge in a military-institutional register.

**Loss-landscape sharpening.** This is the most concrete causal finding available: sharper loss landscapes after continual tuning directly track larger capability drops. On Llama2-7B tuned sequentially across three datasets, MMLU fell by 7.1 points then 17.2 points as landscape sharpness metrics rose in lockstep ([Revisiting Catastrophic Forgetting in LLM Tuning](https://arxiv.org/html/2406.04836v1)). The same study found forgetting severity *increases with model scale* — a 1.1B model lost 0.52% domain knowledge under identical training, while a 13B model lost 41.37% — meaning a larger, more capable base model (which you'd want for nuanced legal/administrative reasoning across French, Arabic, and Darija) is also more fragile ([arXiv:2406.04836](https://arxiv.org/html/2406.04836v1)).

**Parameter interference.** LoRA's low-rank updates can directly collide with the dominant singular directions of the frozen weight matrix that encode essential pretrained knowledge — a phenomenon quantified and named in the OPLoRA paper ([OPLoRA](https://arxiv.org/abs/2510.13003)). A companion scaling-law study found something important for planning: forgetting follows a power law in the number of fine-tuned parameters and update steps, and critically, it **cannot be avoided by early stopping or reducing fine-tuned parameters** — both levers just trade task performance for forgetting rather than eliminating the tradeoff ([Scaling Laws for Forgetting](https://arxiv.org/abs/2401.05605)).

**The stability gap.** Continued pretraining exhibits a well-documented dynamic: a sharp *drop* in performance right at the start of adaptation, before genuine domain gains accrue — a pattern first seen in vision models and now confirmed for LLMs undergoing domain-specific CPT ([Efficient Continual Pre-training by Mitigating the Stability Gap](https://arxiv.org/abs/2406.14833)). Practically, this means the first phase of a FAR-corpus CPT run will look like it's failing — general reasoning and even French/Arabic fluency may visibly degrade — before genuine institutional-domain gains appear. Naive fixed-budget training wastes compute inside this dip.

**Not all forgetting is real.** A useful counterpoint: some measured "forgetting" is actually a drop in task alignment rather than true erasure of knowledge — the model still has the information but has lost the behavioral hooks to surface it. Freezing bottom layers recovered much of this "spurious forgetting" across four continual-learning scenarios ([Spurious Forgetting in Continual Learning](https://arxiv.org/abs/2501.13453)), meaning the diagnosis (is my model forgetting how the civil-service statute cross-references military law, or just forgetting how to answer that specific phrasing of the question) matters before choosing a fix.

---

## 2. What Actually Works — With Numbers

| Technique | Mechanism | Real-world effectiveness | Key limitation |
|---|---|---|---|
| LoRA / QLoRA | Freezes base weights, trains low-rank adapters | Reduces compute cost but does **not** solve the forgetting/performance tradeoff — same inverse power-law relationship as full fine-tuning ([arXiv:2401.05605](https://arxiv.org/abs/2401.05605)); also the technique most consistently recommended in the security literature to limit verbatim memorization of sensitive regulatory/personnel text (see Section 4) | QLoRA costs ~39% more training time for memory savings ([r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1ilkamr/a_comprehensive_overview_of_everything_i_know/)); LoRA needs 5x more tokens than full fine-tuning to match quality in continued-pretraining regimes (20B vs. 4B tokens) ([LoRA Learns Less and Forgets Less](https://www.alphaxiv.org/overview/2405.09673v2)) |
| Elastic Weight Consolidation (EWC) | Fisher-Information-weighted penalty on important parameters | Reduced forgetting up to 17.58% in domain-adapted vision-language models across five datasets ([PA-EWC](https://arxiv.org/abs/2511.20732)); preserved English while adding a new language (Lithuanian) in full-parameter CPT of Gemma2 — directly analogous to preserving general capability while adding French/Arabic/Darija institutional fluency ([EWC for Gemma2](https://arxiv.org/abs/2505.05946)) | Reported to underperform at LLM scale specifically because LLM weights are "polysemantic" — the same weight encodes many unrelated concepts, so protecting it protects noise along with signal ([From Weights to Features](https://www.semanticscholar.org/paper/0b36e24247e8689b19a13237d3d21efb5c5c21a7)) |
| Replay buffers / data mixing | Interleaves general data with domain data at a tuned ratio | As little as 1-5% replay can substantially preserve prior capability; 10-30% typically fully recovers source-validation loss ([D-CPT Law](https://arxiv.org/abs/2406.01375)); one case study hit meaningfully higher domain task performance using only 40% of the original compute budget via smart mixing ([Efficient CPT / Stability Gap](https://arxiv.org/abs/2406.14833)) | Optimal ratio is domain- and scale-dependent, requiring either expensive grid search or scaling-law fitting; requires access to representative general data, which teams fine-tuning third-party checkpoints often lack ([Improved SFT to Mitigate CF](https://arxiv.org/abs/2506.09428)) |
| O-LoRA (orthogonal LoRA) | Constrains sequential task adapters into mutually orthogonal subspaces | Effectively alleviates forgetting under continual instruction tuning versus plain LoRA, no replay storage needed — useful if training separate adapters per branch (Armée Royale, Marine Royale, Gendarmerie) sequentially ([O-LoRA](https://arxiv.org/abs/2310.14152)) | Sensitive to hyperparameters — not uniformly robust ([O-LoRA replication study](https://soapubs.com/index.php/AIDT/article/view/1380)); orthogonal capacity can run out across many sequential domains |
| SLoRA (selective LoRA) | Post-hoc filters LoRA weight updates that cause forgetting, retaining only task-improving changes | 29% less forgetting than standard LoRA on single-task fine-tuning at equivalent task performance ([ACL 2026](https://aclanthology.org/2026.acl-long.513/)); no replay buffer needed, single-pass filter | Requires access to a held-out general-capability validation set to compute the filter — not suitable if no general-data baseline exists; filter strength is a new hyperparameter to tune |
| Model merging (TIES, DARE, Task Arithmetic, SLERP) | Combines independently fine-tuned weight deltas post-hoc | TIES/Localize-and-Stitch sparsification best preserves base knowledge (91.7/89.7 retention scores in MergeBench); linear model averaging outperformed continual-learning regularizers, replay, and LoRA for the RLHF alignment tax specifically ([MergeBench](https://arxiv.org/pdf/2505.10833v2.pdf); [Alignment Tax of RLHF](https://aclanthology.org/2024.emnlp-main.35.pdf)) | DARE's random weight-dropping degrades knowledge retention worse than sparsification methods; merging recovers only ~80% of full-fine-tune performance on small (2-3B) models versus over 90% on 8B+ models |
| Sharpness-Aware Minimization (SAM) | Directly optimizes for flat loss minima | Reversed forgetting into net gains on Llama2-13B (+9.78 relative score change); Llama2-7B ShareGPT run flipped from -6.08 to +5.71 ([arXiv:2406.04836](https://arxiv.org/html/2406.04836v1)) | Roughly doubles compute per step (two forward-backward passes); showed a slight *negative* effect on very small (1.1B) models |

The single most important practical implication: **there is no technique that eliminates forgetting — every method is a tradeoff dial, not an off switch.** Combining replay + LoRA is the pragmatic default; adding EWC or SAM on top is worthwhile only once you have budget for the added complexity and compute. This is compounded, for the FAR use case specifically, by the fact that LoRA's memorization-limiting property (Section 4) makes it doubly attractive here — it is simultaneously the best default against forgetting and the best default against leaking sensitive regulatory or personnel text verbatim.

### MoE-specific training constraints for Qwen3-VL-30B-A3B

Training a Mixture-of-Experts model with LoRA introduces constraints absent in dense models:

- **DeepSpeed ZeRO-2 is required, ZeRO-3 is broken.** ZeRO-3 shards parameters, gradients, and optimizer states across GPUs, but MoE routers route tokens to specific experts on specific GPUs. This creates a `RuntimeError: element 0 of tensors does not require grad` because ZeRO-3's gradient sharding cannot track which expert parameters were actually used by which token. ZeRO-2 (gradient sharding only) works correctly. Confirmed by multiple published guides for Qwen3-30B-A3B LoRA training ([Shaaf Salman, 2025](https://medium.com/@ishaafsalman/fine-tuning-qwen-qwen3-vl-30b-a3b-moe-architecture-with-lora-2365359e870f)).
- **Router layer must not be fine-tuned.** The router determines which 8 of 128 experts activate per token. Fine-tuning the router on FAR-specific data would bias expert selection toward FAR tokens and away from the general distribution, accelerating forgetting. Unsloth disables router fine-tuning by default for this reason.
- **QLoRA NF4 is NOT supported for MoE models.** Bitsandbytes does not support 4-bit quantization on MoE `nn.Parameter` layers, making QLoRA NF4 training impossible on Qwen3-VL-30B-A3B ([Unsloth MoE page](https://unsloth.ai/docs/basics/faster-moe)). The 17.5 GB QLoRA figure cited in some guides applies to the **text-only** Qwen3-30B-A3B, not the VL variant. The Unsloth model card confirms the VL 30B has no 4-bit BnB release. **Use BF16 LoRA instead** (~63 GB on one H200, 78 GB headroom). The recommended framework is Unsloth (not Axolotl, not TRL alone) because of native FastModel support, 2× training speed over Hugging Face trainer, and correct MoE router handling.
- **LoRA target modules** for this model: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. This covers both self-attention and FFN projections. The 8 active experts' FFN weights receive LoRA updates; the 120 inactive experts per token are not updated. Total trainable params: ~53M (0.17% of 31B).
- **Training framework options ranked:**

| Framework | MoE support | Speed | Notes |
|---|---|---|---|
| Unsloth (FastModel) | Native, router frozen by default | 2× HF trainer | **Recommended** — verified Qwen3-30B-A3B support |
| Qwen3-VL official finetune | Via scripts/sft_30a3b_lora.sh | Standard | Works but less optimized |
| TRL SFTTrainer | Requires manual ZeRO-2 config | Standard | Possible but more setup needed |
| Axolotl | Qwen3 config available | Standard | Alternative, no specific advantage |
| LitGPT | No MoE support | N/A | Not usable |

---



## 3. Institutional/Legal Domain Adaptation: Mixed, Not Uniformly Positive Evidence

This is the finding most directly relevant to a FAR institutional-expert project, and it is a genuine caution, mirroring what a comparable JAMIA-style study found for medicine. A 2025 legal-domain benchmarking effort found that general-purpose LLMs applied to legal queries hallucinate in **approximately 58% to 82%** of cases even without any domain fine-tuning, but customized/fine-tuned legal AI systems still hallucinate at **17% to 33%** — "undermining claims of legal technology providers about legal research tools being substantially less prone to hallucination" ([Fine-grained Claim-level RAG Benchmark for Law, arXiv](https://arxiv.org/html/2605.21071v3), citing Dahl et al. 2024 and Magesh et al. 2025). The same paper is explicit that hallucination is a near-fundamental property of LLMs "irrespective of their architecture, training data quality, or scale," which is a materially different conclusion from "fine-tuning solves this" — it means domain fine-tuning shifts the failure rate downward but does not remove the underlying risk class. Legal citation-prediction benchmarking independently confirms that "neither general nor law-specialised LLMs suffice as stand-alone solutions, with performance near zero" without retrieval ([arXiv 2412.06272](https://arxiv.org/abs/2412.06272)).

Against that caution, the success stories in adjacent domains share a common thread — careful staging, domain-partitioned retrieval, and compute-efficient design rather than brute-force fine-tuning on the raw corpus:

- **Domain-partitioned hybrid RAG + knowledge graph** for legal research achieved a **70% pass rate versus 37.5% for RAG-only** on a comparable benchmark, evidence that structuring retrieval around the corpus's actual legal hierarchy (chapters, articles, cross-references) — not deeper fine-tuning — was the decisive lever ([arXiv 2602.23371](https://arxiv.org/abs/2602.23371)).
- **ChatLaw**, fine-tuned on Chinese legal datasets with an added self-attention mechanism, demonstrated a significant reduction in hallucination versus generalized LLMs, showing targeted domain fine-tuning *can* help — but notably as an enhancement layered onto a still-retrieval-grounded system, not a replacement for grounding ([legal LLM hallucination review](https://arxiv.org/html/2605.21071v3)).
- **Commercial RAG legal tools** (Lexis+ AI, Westlaw AI-Assisted Research, Ask Practical Law) cut hallucination to 17-34% — a real improvement over the ~58-82% ungrounded baseline, but this gain came from retrieval architecture and citation-verification tooling, not from deeper fine-tuning of the underlying model ([Stanford "Hallucination-Free?"](https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf)).

The pattern across both the negative and positive findings mirrors the biomedical case: naive "just fine-tune on the regulatory corpus" produces a disappointing, still-double-digit-hallucination result, while staged, retrieval-augmented, hierarchy-aware approaches produce the success stories. For a single-institution model like this one, this argues strongly for a RAG-plus-light-fine-tuning hybrid architecture rather than SFT-only — fine-tuning contributes tone, terminology, and reasoning format; RAG contributes the actual grounded facts about FAR regulations, structure, and law.

### Multi-agent LLM-KE extraction for document-to-knowledge-base pipeline

Converting ~125 GB of raw PDFs, DOCXs, and MSGs into structured training data requires more than naive chunk-and-embed. The proven approach for military-domain knowledge graph construction — where crowdsourcing is impossible (security constraints) — is **multi-agent LLM extraction with domain expert roles** (Tao et al., 2026, *Computers, Materials & Continua*, [PDF](https://cdn.techscience.press/files/cmc/2025/TSP_CMC-86-1/TSP_CMC_68670/TSP_CMC_68670.pdf)):

| Agent role | Extracts | Prompt type |
|---|---|---|
| Command expert | Entities: Général Berrid, DAG, 2e GMR. Relations: COMMANDE, RELEVE_DE | Few-shot NER |
| Legal expert | Entities: Dahir 1-74-383, Article 63 bis. Relations: REGIT_PAR, CITE | Few-shot + ontology |
| HR admin | Entities: pension taux, grade, avancement. Relations: APPLIQUE_A | Few-shot |
| Equipment specialist | Entities: M1A1 Abrams, FREMM, F-16. Relations: EQUIPE, DEPLOIE | Few-shot |

**Key finding:** LLM-KE achieves F1 comparable to supervised models (78%+) using zero-shot LLM prompts, not fine-tuning. The ontology is dynamically updated via hierarchical clustering consistency checks. This means the multi-agent extraction step does NOT require its own fine-tuned models — off-the-shelf frontier LLMs with well-structured few-shot prompts are sufficient for the extraction pipeline. The resulting triples populate a Qdrant vector store (FAR knowledge wiki), which is then used to generate structured SFT/RLHF training pairs.

### Target language distribution for generated SFT data

FAR operates in three languages with a fourth code-switched mode. The training data should reflect real usage, not equal weighting:

| Language | Share | Rationale |
|---|---|---|
| French | 55% | Primary administrative and legal language; dahrirs, decrees, internal memos |
| Darija (Moroccan Arabic dialect) | 30% | Day-to-day operational communication, oral orders, informal HR queries |
| Modern Standard Arabic | 10% | Formal correspondence, King's speeches, BO publications |
| Code-switched | 5% | Mixed French/Darija/Arabic — the most realistic mode for actual soldier-administrator interaction |

**Darija handling requires deliberate attention.** Darija has no standardized orthography; it is primarily spoken with ad-hoc Latin/Arabic-script transcription. Research from ACL 2025 ArabicNLP shows that separating Darija data into its own labeled category improves F1 by +30 points over treating it as "Arabic" ([Falcon3-7B-Arabic](https://arxiv.org/abs/2505.08015)). The training pipeline should label Darija examples explicitly in the chat template.

---

## 4. Data Curation Problems Specific to the FAR Institutional Corpus

### Licensing and access restrictions are real, not theoretical, blockers

Unlike a purely academic domain, a significant portion of the ideal training corpus here is not publicly available at all. The Discipline Générale Militaire's full annexed text (Dahir n° 1-74-383) exists only in secondary-source summaries and article-level cross-references in open sources — no complete, official government-hosted version was located, meaning the single most foundational document for this project has an access gap before any fine-tuning question even arises ([Scribd dahir text](https://www.scribd.com/document/465957465/reglement-de-discipline-generale-des-forces-armees-royales-3-pdf)). Internal Chief-of-Staff implementing directives (referenced by Article 3 of the same dahir) are explicitly non-public by design — Article 3 delegates operational detail to internal decisions never intended for the *Bulletin Officiel*. This creates a licensing/access profile closer to a restricted-access legal database than an open academic corpus: the best-governed, most authoritative institutional knowledge is the hardest to legally and practically acquire for training, and what remains freely available (public dahirs, décrets, published statutes) is necessarily the shallower, more general layer of the institution's actual operating knowledge — a direct parallel to the organism-domain finding that the best-curated species databases carry the most restrictive licenses.

**Known public source URLs for FAR legal documents (current as of mid-2026):**

| Document | Source URL |
|---|---|
| DGM (Discipline Générale Militaire, Dahir 1-74-383) | [Scribd](https://www.scribd.com/document/465957465/reglement-de-discipline-generale-des-forces-armees-royales-3-pdf) (secondary scan, no official host) |
| Code de Justice Militaire (Dahir 1-56-270) | [Menarights.org](https://menarights.org/en/document/dahir-ndeg-1-56-270-10-november-1956-concerning-military-justice) (English translation) |
| Pensions Militaires (Loi 013-71) | [Acaps.ma](https://acaps.ma/loi-013-71/) (CMR regime overview) |
| Statut Général de la Fonction Publique (Dahir 1-58-008) | [Mmsp.gov.ma](https://www.mmsp.gov.ma/fr/statut-general-de-la-fonction-publique) |
| FAR Overview / UN Fact Sheet | [UN Peacekeeping](https://peacekeeping.un.org/en/morocco) |

FAR's organizational structure is inherently graph-like, not prose-like: a hierarchy of King → Ministre délégué → Inspecteur Général → branch commands → Direction des Affaires Générales → its four divisions, cross-cut by a separate legal hierarchy of dahirs → décrets → statutes → internal directives, with explicit cross-references between them (e.g., the officer statute's Article 63 bis referencing civil-service law, or Loi 44-18 explicitly subjecting conscripts to three separate legal instruments simultaneously). Verbalizing this into flat training text risks exactly the same information loss documented when converting biological ontologies (KEGG, Reactome, GO) into narrative training text — general-purpose LLM narration of structured graphs has been shown to accurately capture only around two-thirds of relationships even with strong LLM assistance ([GPTON, on ontology verbalization](https://arxiv.org/abs/2410.10899)). The safer architectural choice, mirroring the KRAGEN/ESCARGOT pattern in the biology literature, is to keep the organizational chart and legal cross-reference graph in native structured form (a knowledge graph or relational schema) and query it via retrieval at inference time, rather than attempting to bake the full hierarchy into model weights through fine-tuning.

### Tokenization breaks on legal-citation notation, not just biological notation

Generic BPE/WordPiece tokenizers are built for natural-language statistics and transfer poorly to any text with dense, structured internal notation — this applies to legal citation numbering (dahir numbers like "1-74-383," article/paragraph references, cross-reference chains like "sous réserve de l'article 12 ci-dessus") just as much as it does to protein sequences or SMILES chemical strings. A generic tokenizer has no inherent reason to treat "Article 86" as a stable, atomic reference to a specific legal provision rather than as arbitrary sub-tokens, which directly compounds the citation-hallucination risk discussed in Section 5 — the model is not just uncertain about *what* Article 86 says, but its tokenizer may not even represent article numbers as consistent, comparable units across the corpus. If domain-specific tokens (article-number patterns, dahir-numbering conventions, DAG-specific abbreviations) are added to the tokenizer, naive random initialization of their embeddings actively destabilizes early fine-tuning — the first gradient steps "just remove probability from the new words" — with embedding-averaging shown as a better default ([Columbia CS: Vocabulary Expansion](https://www.cs.columbia.edu/~johnhew/vocab-expansion.html)). This problem compounds further across the trilingual French/Arabic/Darija surface this system needs to operate in, since Arabic morphology and unstandardized Darija orthography each interact with tokenization differently than French legal prose.

### Vocabulary expansion is validated by recent research — target +10K-15K FAR-specific tokens

Two recent papers confirm that expanding LLM vocabulary with domain-specific tokens is effective if done correctly, and the Qwen3 tokenizer (~152K tokens) has headroom for additions:

- **EACL 2026 Findings** (Purason et al.): Continued BPE training (not appending tokens) ensures new tokens are reachable by the tokenizer. New-token adoption rate reaches ~98% after 10K continued pretraining steps. The key technique is training the BPE tokenizer further on the domain corpus (not hardcoding new tokens), so existing sub-token merges that already approximate the new token naturally rebalance ([EACL 2026](https://aclanthology.org/2026.findings-eacl.341/)).
- **NeurIPS 2025** (Herold et al.): Adding +30K domain tokens to LLaMA 3.1 produced 20% shorter sequences and 20-30% inference speedup. Model quality fully preserved on both general and domain benchmarks. Embedding initialization via averaging of constituent sub-token embeddings, not random initialization ([arxiv 2509.26124](https://arxiv.org/abs/2509.26124)).
- **OPLoRA overlap caution:** Vocabulary expansion increases the embedding matrix size (152K → ~165K tokens). This adds ~9 GB to the BF16 base model's embedding layer. If LoRA target modules include `embed_tokens`, the expanded embeddings receive LoRA updates that can collide with the orthogonal constraints described in Section 2's O-LoRA discussion.

**Target tokens for FAR (estimated ~10K-15K new tokens):**

| Category | Examples | Count (est.) |
|---|---|---|
| Dahir/décret numbers | `1-74-383`, `44-18`, `2-23-925` | ~500 |
| FAR acronyms | `DGM`, `DAG`, `DGSN`, `TSD`, `BSN`, `BSP`, `DHR`, `DAJ`, `DBF` | ~200 |
| French legal phrases | `nonobstant`, `sous réserve de`, `vu la loi`, `attendu que` | ~1,000 |
| Arabic administrative | `القوات المسلحة الملكية`, `المجند`, `الخدمة العسكرية` | ~3,000 |
| Darija administrative | `l'moul'`, `l'khdma`, `l'prime`, `l'congé`, `l'avancement` | ~2,000 |
| Rank/unit abbreviations | `GMR`, `BSR`, `EMG`, `CMD`, `COL`, `GAL` | ~300 |
| Compound legal cross-refs | `art63bis-L44-18`, `DM-1-74-383-art12` | ~500 |

**CAREFUL:** Only add tokens that appear **>100 times** in the corpus. Adding low-frequency tokens bloats the embedding matrix without measurable benefit. The Qwen3 tokenizer is already large (~152K) — each added token costs embedding dimension × 2 bytes (BF16) × 2 (input + output embedding) ≈ 2048 × 2 × 2 = 8 KB per token.

**Implementation sequence:** continued BPE on FAR corpus → embedding averaging init → 10K CPT steps with replay → verify cross-entropy loss per language (French/Arabic/Darija) before proceeding to SFT.

### Data is extremely skewed toward the parts of the institution that are publicly documented

Just as ten bacterial species account for half of all bacteriology literature while most species remain unstudied, FAR's public documentary record is heavily skewed toward its most publicly visible instruments: the founding dahirs, published décrets, pension law, and high-level organizational facts are comparatively well-documented, while day-to-day administrative procedure, unit-level directives, and the actual operational texture of HR casework are almost entirely absent from any source that could legally be assembled into a training corpus. Any FAR-specific corpus built from open sources inherits this skew directly — the model will appear confident and well-grounded on the topics that happen to be publicly documented (pension formulas, high-level rank structure) while having essentially no real training signal on the topics soldiers and administrators most often need help with (unit-level leave procedures, specific case-handling workflows), which is precisely the kind of skew that standard class-imbalance mitigations (oversampling, weighted loss) cannot compensate for when the underlying problem is absolute data scarcity, not just class ratio.

### Synthetic data generation carries specific risks in this domain

Given the data-scarcity problem above, generating synthetic training examples with a stronger LLM (e.g., asking a frontier model to generate plausible HR-question/answer pairs grounded in known FAR structure) is a natural mitigation — but it inherits and can amplify the generator's biases across multiple bias types ([Bias Inheritance in LLM Data Augmentation](https://arxiv.org/abs/2502.04419)), and uniform synthetic-data formatting can cause "pattern overfitting" that shifts downstream model behavior relative to organic data ([Synthetic Data Flaws](https://arxiv.org/abs/2406.12397)). More concerning for factuality in this specific application: a "knowledge mismatch" between what synthetic training data assumes the model knows and what it actually knows is empirically linked to increased hallucination on unseen data ([Knowledge Mismatch Hypothesis](https://arxiv.org/abs/2411.00878)) — a frontier model asked to generate FAR-HR training examples will confidently invent plausible-sounding but non-existent internal procedures precisely because it has no real grounding in FAR's actual (non-public) administrative practice, and that fabricated confidence would then be distilled directly into the fine-tuned model's training signal.

---

## 5. The Hallucination Paradox: Fine-Tuning Can Make Things Worse, Not Better

This is the least intuitive and most important finding for this project, and it carries particular weight for an institution where a wrong answer about disciplinary procedure, pension eligibility, or promotion criteria has direct real-world consequences for a service member's career and finances. The core mechanism comes from a controlled study (Gekhman et al.) that varied how much new factual knowledge (not present in pretraining) appeared in fine-tuning examples. The result: LLMs learn examples introducing genuinely new knowledge *slower* than familiar ones, but once learned, those new-knowledge examples **linearly increase the model's tendency to hallucinate** ([Does Fine-Tuning Encourage Hallucinations?](https://arxiv.org/abs/2405.05904)). The paper's interpretation is stark — LLMs mostly acquire factual knowledge during pretraining; fine-tuning teaches them to use existing knowledge more efficiently, not to reliably absorb new facts. Teaching a base model genuinely novel, specific institutional facts — an exact article number, a specific DAG division's mandate, a precise pension-rate formula — is, by this framing, close to a best case for triggering exactly this failure mode, since almost none of this content exists in any base model's pretraining data.

This is compounded by an overconfidence effect: instruction fine-tuning systematically pushes models to generate "long-tail knowledge not well covered during pretraining," making them more informative but measurably less truthful on unseen tasks ([Balancing Truthfulness and Informativeness](https://arxiv.org/abs/2502.11962)). Fine-tuned models trained on small, specialized datasets — exactly what a FAR-specific corpus would be, given the scarcity documented in Section 4 — "often exhibit overconfidence... resulting in poor calibration" ([Calibrating LLMs](https://arxiv.org/abs/2503.05777)).

The documented rates in the legal domain specifically are sobering and directly transferable: general-purpose LLMs applied to legal queries hallucinate in **58% to 82%** of cases without grounding ([legal RAG benchmark citing Dahl et al. 2024](https://arxiv.org/html/2605.21071v3)); even the best commercial RAG-grounded legal research tools still hallucinate in **17% to 33%** of cases ([Stanford "Hallucination-Free?"](https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf)); labor/HR-adjacent law has been specifically flagged as a higher-hallucination-risk subdomain relative to areas like constitutional law, which is precisely the FAR use case ([legal hallucination domain-variance study](https://arxiv.org/pdf/2606.00898.pdf)). Real-world consequences of this class of failure are not hypothetical: a Canadian tribunal held Air Canada liable for its chatbot's incorrect bereavement-fare policy statement, explicitly rejecting the airline's argument that the chatbot was a separate responsible entity ([BBC](https://www.bbc.com/travel/article/20240222-air-canada-chatbot-misinformation-what-travellers-should-know)); New York City's government chatbot gave business owners answers that would have led them to violate labor law, including citing a superseded minimum-wage figure ([Reuters](https://www.reuters.com/technology/new-york-city-defends-ai-chatbot-that-advised-entrepreneurs-break-laws-2024-04-04/)); and a Deloitte Australia government report containing AI-fabricated citations led to a partial refund of a roughly $190,000 contract ([Fortune](https://fortune.com/2025/10/07/deloitte-ai-australia-government-report-hallucinations-technology-290000-refund/)). Each case shares the same signature relevant here: fluent, confident, plausible-sounding incorrect institutional/policy information, not caught until external discovery.

Counterintuitively, **fine-tuned domain models are not reliably better at catching their own hallucinations** than general-purpose models with strong reasoning — a pattern seen consistently across biomedical benchmarks (MedHallu, medical hallucination-probing studies) where general-purpose models with chain-of-thought prompting outperformed domain-fine-tuned counterparts at hallucination *detection* specifically ([MedHallu](https://aclanthology.org/2025.emnlp-main.143/); [Medical Hallucination in Foundation Models](https://arxiv.org/html/2503.05777v2)). This is a genuine architectural signal directly applicable to the FAR case: reasoning and retrieval may do more for factuality than deep SFT on the regulatory corpus itself, reinforcing the Section 3 conclusion that RAG-plus-light-fine-tuning, not SFT-heavy adaptation, is the defensible target architecture.

**Mitigations that actually reduce this risk:** retrieval-augmented generation grounds generation in verifiable external sources (the actual dahir/décret text, retrieved and cited) rather than relying purely on parametric memory, though it "does not guarantee valid responses if retrieval fails" ([CONFLARE](https://arxiv.org/abs/2404.04287)); domain-partitioned retrieval architectures that respect the corpus's actual legal hierarchy achieved a 70% pass rate versus 37.5% for flat RAG ([arXiv 2602.23371](https://arxiv.org/abs/2602.23371)); and always surfacing the retrieved source passage to the end user (rather than only a generated summary) gives a human verification point that none of the Air Canada, NYC, or Deloitte failures had in place. The Legal RAG Bench (Feb 2026) confirms that most legal hallucination failures originate in retrieval — the model can answer correctly from the right passage, but retrieval fails to surface it — and that domain-adapted embeddings (Kanon 2) recover +17 points accuracy over general-purpose embeddings on legal retrieval tasks. This makes embedding quality a first-order decision for the FAR pipeline, not just a RAG tuning detail.

**New mitigations (2025–2026):** self-distillation during SFT reduces hallucination rates from ~15% to ~3% by training from the model's own generated rationales rather than gold-only labels ([Kaplan et al. 2026, arXiv 2604.15574](https://arxiv.org/abs/2604.15574)); TRL's `SDFTTrainer` now implements this for Hugging Face pipelines. For retrieval failures specifically, the KnownPatch method at inference time detects when retrieved context is irrelevant and triggers a controlled refusal rather than a speculative answer ([arXiv 2406.13214](https://arxiv.org/abs/2406.13214)). Applying both in the FAR pipeline — SDFT during training, KnownPatch at inference — covers the two orthogonal failure modes: parametric hallucination (SDFT) and retrieval-gap hallucination (KnownPatch).

---

## 6. Evaluation Is Harder Than It Looks

### Standard benchmarks are compromised, not just imperfect

MMLU shows roughly 29.1% contamination signal by one estimate ([JHU/NAACL 2024](https://blog.pebblous.ai/blog/llm-benchmark-contamination/en/)), and manual re-annotation found a 6.49% overall ground-truth error rate that spikes sharply in specific subject subsets — the Virology subset alone showed a 57% error rate in one audit ([Are We Done with MMLU?](https://arxiv.org/html/2406.04127v3)), a useful cautionary parallel for the MMLU "Professional Law" and "International Law" subsets that would be the most obviously relevant off-the-shelf benchmarks for a FAR project — these general legal-knowledge subsets were built for common-law, largely US-centric legal reasoning and provide essentially no signal on Moroccan civil-law structure, French-language legal drafting conventions, or the FAR's own internal disciplinary regime. None of this measures FAR-specific institutional competence in the first place, but it establishes that even "passing" general legal benchmarks would be a far weaker signal than it appears for this project's actual goals.

### Legal-domain benchmarks reveal how shallow current institutional understanding is

Purpose-built legal benchmarks such as LegalBench (evaluating legal reasoning inspired by American legal reasoning) and LEXTREME (classification/NER tasks) exist specifically because general-purpose benchmarks proved inadequate for legal competence assessment ([legal RAG benchmark survey](https://arxiv.org/html/2605.21071v3)) — but neither was built with Moroccan military-administrative law in mind, and no equivalent FAR-specific or even general Moroccan-legal benchmark exists in the public literature. This is a direct warning, mirroring the organism-domain finding that general biological fluency does not imply true biological grounding: a model's fluency in general legal reasoning or even general Arabic/French legal terminology does not imply grounding in the FAR's specific, non-public administrative reality, and instruction tuning alone, without a custom evaluation set built for this exact institution, would leave that gap entirely unmeasured.

### Multi-hop institutional reasoning is a real risk surface

Just as biological multi-hop causal reasoning ("what happens downstream if gene X is knocked out") collapses sharply for even strong LLMs, multi-hop institutional/legal reasoning has an exact analogue here: "if a conscript under Loi 44-18 commits an infraction, which of the three applicable legal instruments (Discipline Générale Militaire, Code de Justice Militaire, Loi 01-12) governs, and does the civil-service leave provision under Article 13 still apply during any resulting arrêts de forteresse period?" is a genuinely multi-hop question requiring the model to correctly traverse cross-references across three to four separate legal instruments simultaneously. General multi-hop reasoning benchmarks in adjacent domains show sharp precision collapse from one hop to two hops even in frontier models ([BioHopR-style multi-hop degradation pattern](https://aclanthology.org/2025.findings-acl.668.pdf)), and there is no reason to expect legal cross-reference reasoning to be structurally easier — if anything, the explicit, non-narrative cross-referencing style of legal drafting (versus implicit causal chains in biology) may make it harder for a model trained mostly on narrative text to track correctly.

### Expert annotation is expensive and only moderately reliable — and harder still for non-public content

Specialized legal/institutional annotation requires either practicing Moroccan military-legal experts or officers with direct FAR administrative experience, a scarcer and more expensive annotator pool than general crowd annotation, and — critically — much of what would need to be annotated as "ground truth" (actual case outcomes, internal directive interpretations) is not publicly documented at all, meaning the annotation process itself may require access to the same restricted internal material discussed in Section 4. Complex legal/institutional judgments are also inherently more contested than they first appear: even well-resourced clinical/legal annotation efforts elsewhere report only "moderate" inter-annotator agreement on complex judgment calls, underscoring that the "ground truth" against which a FAR-expert model's multi-hop reasoning would be validated is itself difficult to establish with confidence, not a fixed target waiting to be measured against.

### LLM-as-judge has documented reliability problems, and shares the same domain gap

GPT-4 as judge shows measurable self-preference bias, favoring text with lower perplexity relative to itself regardless of actual quality ([Self-Preference Bias](https://arxiv.org/html/2410.21819v1)), and judge models share the same domain-knowledge gaps as the models they're judging — MMLU-Pro error analysis attributed 35% of GPT-4o's errors specifically to lack of domain expertise ([MMLU-Pro](https://proceedings.neurips.cc/paper_files/paper/2024/file/ad236edc564f3e3156e1b2feafb99a24-Paper-Datasets_and_Benchmarks_Track.pdf)). Using a general frontier model as an automated judge for FAR-specific institutional answers inherits exactly this risk: the judge model has no more real grounding in FAR's non-public administrative reality than the model being evaluated, so LLM-as-judge should be treated as a cheap first-pass filter at best, not a substitute for expert (ideally FAR-affiliated) human review before any evaluation result is trusted.

### "General capability retention" has no agreed metric

The TRACE benchmark exists specifically because standard continual-learning benchmarks were too easy for aligned LLMs; on its harder task sequence, Llama2-13B's GSM8K accuracy collapsed sharply after modest domain-specific tuning ([TRACE](https://openreview.net/pdf?id=3qa4YLkcEw)) — an illustration of how abrupt and severe forgetting can be even outside worst-case settings, and a direct warning for a FAR project that a model fine-tuned deeply enough to reliably recite Discipline Générale Militaire articles could simultaneously lose general reasoning capability needed for open-ended HR conversation, without that tradeoff showing up on any single benchmark score. There is no single scalar that reliably captures "did I keep the model's general reasoning intact while adding institutional depth" — practitioners must track multiple benchmark deltas (general reasoning, French/Arabic/Darija fluency, legal citation accuracy) separately and accept that averaging across heterogeneous tasks hides which specific capability degraded.

### Available military-domain evaluation benchmarks (2025-2026)

Three benchmarks exist in the literature for evaluating LLMs in military/defense contexts, though none test Moroccan-specific law:

| Benchmark | Source | What it tests | Size | Directly applicable? |
|---|---|---|---|---|
| **ARMOR 2025** | Virginia Tech, [arXiv 2605.00245](https://arxiv.org/abs/2605.00245) | Law of War, ROE, Joint Ethics (OODA taxonomy) | 519 prompts, 12 categories | Partial — doctrinal scenarios are transferable |
| **WARBENCH** | [arXiv 2603.21280](https://arxiv.org/abs/2603.21280) | IHL compliance, targeting decisions, proportionality | 136 scenarios from real conflicts | Partial — IHL is universal, but US-centric |
| **MEBL** | IEEE Dataport (Gupta, 2026) | 4 axes: foundational, mission, robustness, resource | Multi-criteria scoring | Framework is adaptable |

### Custom FAR evaluation set is required (no existing benchmark covers Moroccan military law)

The document's Section 6 is explicit: there is no existing benchmark that measures FAR-specific institutional competence, and no general Moroccan-legal benchmark to fall back on. A custom evaluation set must be built:

| Category | # questions | Source | Metric |
|---|---|---|---|
| DGM article exact recall | 200 | From wiki, exact match | Precision@1 |
| Multi-hop cross-reference | 150 | Art X × Art Y chains | F1 over cited articles |
| French fluency | 100 | Holdout admin text perplexity | Perplexity |
| Darija fluency | 100 | Holdout Darija perplexity | Perplexity |
| General reasoning (MMLU subset) | 200 | Track forgetting | Delta from baseline |
| ARMOR-style doctrinal | 200 | Adapted from ARMOR taxonomy to FAR context | Pass rate |

**Recommended evaluation tools:**

- **DeepEval** (pytest-style, CI-integrated) — best for automated factual precision measurement. Supports custom metrics, RAGAS for retrieval quality, and hallucination detection.
- **LangSmith** — good for online evaluation against production traces once the system is deployed.
- **LLM-as-judge** — use only with structured rubrics and FAR-domain calibration. Section 6's warning stands: treat LLM-as-judge as a cheap first-pass filter at best, not a substitute for expert review.

---

## 7. Compute, Cost, and Infrastructure Reality Check

### Hardware is 2× H200 141GB PCIe (no NVLink) — not consumer GPU

The target deployment is **2× NVIDIA H200 141GB** connected via PCIe (no NVLink), running **Pipeline Parallelism PP=2**. This is fundamentally different hardware from the consumer-GPU scenarios commonly discussed in open-source fine-tuning guides, and several constraints follow directly:

- **No Tensor Parallelism** — PP=2 is the only viable option because PCIe bandwidth (~32 GB/s per direction) is ~12× slower than NVLink (~900 GB/s). TP across PCIe would be memory-bound.
- **BF16 base model fits on one GPU** — the BF16 checkpoint of Qwen3-VL-30B-A3B is ~61 GB, well within a single H200's 141 GB. The second GPU runs the other pipeline stage.
- **QLoRA NF4 is NOT available for this model.** Bitsandbytes has no 4-bit support for MoE architectures; QLoRA NF4 training is impossible on Qwen3-VL-30B-A3B ([Unsloth MoE page](https://unsloth.ai/docs/basics/faster-moe)). The ~17.5 GB figure cited in some guides applies to the text-only Qwen3-30B-A3B, not the VL variant. Unsloth's VL model card confirms no 4-bit BnB release exists for 30B MoE.
- **LoRA BF16: ~63 GB** — fits on one H200 with 78 GB headroom (well within 141 GB). This is the correct training configuration. Single-GPU LoRA BF16 is feasible, leaving GPU 2 free for larger batch sizes or longer sequences via pipeline parallelism.
- **DeepSpeed ZeRO-2 is required** — ZeRO-3 breaks LoRA gradient flow on MoE architectures (`RuntimeError: element 0 of tensors does not require grad`). ZeRO-2 with optimizer offload to CPU works correctly.

### VRAM breakdown for inference at 64K context

With the deployed vLLM stack (FP8 KV cache, PP=2, GPU_MEMORY_UTILIZATION=0.95):

| Component | Per GPU | Notes |
|---|---|---|
| Model weights (AWQ INT4) | ~8.5 GB | 17 GB total across 2 GPUs |
| KV cache per user (FP8, 64K) | ~1.5 GB | 4 KV heads × 128 head_dim × 65536 tokens × 1 byte (FP8) ÷ 2 (PP=2) |
| On-GPU capacity | ~84 users | ~126 GB usable (0.95 × 141 GB minus ~8.5 GB weights = ~126 GB) ÷ 1.5 GB |
| CPU swap | ~166 users | Remaining 166 of 250 users served via swap over PCIe |

**FP8 is the best KV cache dtype available on H200 (Hopper).** INT4 and FP4 KV cache are not supported on Hopper architecture — they require Blackwell (B200/B300). The vLLM flag is `--kv-cache-dtype fp8`.

### Training cost and time estimates

For BF16 LoRA on Qwen3-VL-30B-A3B (53M trainable params, 0.17% of 31B total) on 2× H200 with ZeRO-2:

| Dataset size | Steps (3 epochs, GA=8, BS=1) | Time estimate (H200) | Time estimate (8×A100 80GB, baseline) |
|---|---|---|---|
| 5K examples | ~1,875 | ~4 hours | ~8-10 hours |
| 20K examples | ~7,500 | ~16 hours | ~32-40 hours |
| 50K examples | ~18,750 | ~40 hours | ~80-100 hours |

The practical takeaway, consistent with the biomedical finding: the end-to-end cost is dominated by data curation and evaluation-harness work, not raw compute. The BF16 LoRA configuration is by far the most defensible starting point given the FAR corpus's scarcity and access constraints documented in Section 4 — a deep full-SFT or continued-pretraining run is both the most expensive option and, per Section 3, not the option best supported by comparable-domain evidence.

### Training instability is not a bug, it's a documented dynamic

Learning-rate rewarming during CPT reliably produces an initial loss spike on *both* the original and new-domain data before eventual improvement, and this occurs even when continuing training on the exact same distribution — evidence that the instability is driven by optimization dynamics, not just distribution shift ([NeurIPS 2023 CPT workshop paper](https://neurips2023-enlsp.github.io/papers/paper_76.pdf)). Bigger learning rates cause bigger transient spikes but stronger domain adaptation; the practical recommendation is linear warmup plus cosine decay to about 10% of max learning rate, not a constant rate — a relevant caution before assuming a single training run on the FAR corpus will show clean, monotonic improvement.

### Data pipeline engineering is a systems problem, not just an ML problem

Even for a comparatively small FAR corpus (measured in individual legal texts and directives rather than millions of documents), the same toolchain fragility documented at scale applies: Hugging Face TRL's SFTTrainer has had documented, real GitHub issues around streaming-dataset incompatibility, gradient-accumulation loss-scaling bugs, and chat-template application failures on specific model families ([TRL Issue #2138](https://github.com/huggingface/trl/issues/2138); [TRL Issue #3140](https://github.com/huggingface/trl/issues/3140)) — worth knowing before assuming the toolchain "just works" out of the box, particularly in an air-gapped environment where debugging cannot rely on searching for the latest GitHub issue thread mid-project.

### Quantization is broadly safe, but structured/precise QA is more sensitive than average

General-purpose INT4 quantization recovers about 99.36% of BF16 baseline accuracy on average across benchmarks ([FP8/INT8/INT4 Quantization Study](https://arxiv.org/html/2411.02355v1)), but domain-specific quantized models have shown materially larger accuracy drops on precise factual-recall tasks (a 4-bit quantized medical model retained only 85.9% of base accuracy on a precise medical QA benchmark — [MedGemma 4-bit model card](https://huggingface.co/tarirozw/medgemma-1.5-4b-4bit-v1)). Since legal citation accuracy (exact article numbers, exact pension-rate figures) is precisely this kind of precise factual recall, any quantized checkpoint should be benchmarked on a FAR-specific evaluation set directly rather than assumed safe from general-purpose quantization literature.

**Training uses the BF16 base model** (`Qwen/Qwen3-VL-30B-A3B-Instruct`), NOT the AWQ checkpoint. AWQ quantization is a post-training step applied after LoRA merging. The inference stack deploys the AWQ-quantized merged model via vLLM with `--quantization awq`. The pipeline is: train LoRA on BF16 base → merge adapter → AWQ quantize → deploy to vLLM.

---

## 8. Practical Problems Practitioners Actually Report

Drawn from applied retrospectives rather than academic papers, several patterns recur and are directly actionable for a narrow, low-resource institutional domain like this one:

- **Data work dominates the effort budget.** One detailed practitioner retrospective reports roughly 80% of total project effort spent on data (formatting, labeling, deduplication, augmentation), about 15% on evaluation harnesses, and only about 5% on infrastructure/configuration — shifting toward 90/10 once scripts stabilize ([r/LocalLLaMA overview](https://www.reddit.com/r/LocalLLaMA/comments/1ilkamr/a_comprehensive_overview_of_everything_i_know/)). For FAR specifically, this data-work share should be expected to be even higher, given the access-restriction and structured-to-text conversion problems documented in Section 4.
- **Narrow fine-tuning silently strips general world knowledge** not explicitly represented in the fine-tuning set; reported mitigations include merging target data with general open-source datasets, weight-averaging the fine-tuned and base model, and reserving roughly 25% of training data for the primary language/domain plus 5-10% general instructional data ([r/LocalLLaMA on counter-productive fine-tuning](https://www.reddit.com/r/LocalLLaMA/comments/1on2dja/have_you_ever_encountered_a_case_where_finetuning/)) — directly relevant given the risk of degrading the model's general French/Arabic conversational fluency while over-indexing on formal legal-register text.
- **LoRA/QLoRA alone is often insufficient for genuinely narrow domains.** For niche topics with scarce natural text (directly analogous to the FAR corpus's documented scarcity in Section 4), practitioners recommend maximizing trainable rank, heavy data augmentation (phrasing the same institutional fact multiple ways, in multiple languages/registers), and synthetic generation from a stronger model — while being aware of the synthetic-data risks discussed in Section 4 ([r/LocalLLaMA on domain training](https://www.reddit.com/r/LocalLLaMA/comments/1ax040f/trainingfinetuning_local_llm_on_specific_domain/)).
- **You usually can't replicate the base model's original SFT recipe.** Anyone fine-tuning a third-party open checkpoint (Llama, Mistral, Gemma) typically lacks the original instruction-tuning data mixture, making textbook replay-based anti-forgetting approaches inapplicable without reconstructing a synthetic approximation of that original distribution ([Improved SFT to Mitigate Catastrophic Forgetting](https://arxiv.org/abs/2506.09428)).
- **Pruning plus insufficient recovery training is a common failure combination** — don't combine aggressive pruning (e.g., to fit a larger model into a smaller air-gapped inference footprint) with fine-tuning unless there is real compute budget and time set aside for recovery.

### Published working recipe for Qwen3-VL-30B-A3B LoRA training

The following configuration is adapted from Shaaf Salman's published guide (2025, 8× A100 80GB) and adjusted for 2× H200 141GB with ZeRO-2 ([source](https://medium.com/@ishaafsalman/fine-tuning-qwen-qwen3-vl-30b-a3b-moe-architecture-with-lora-2365359e870f)):

**DeepSpeed ZeRO-2 configuration (`ds_config_zero2.json`):**
```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {"device": "cpu"},
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "bf16": {"enabled": true},
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto"
}
```

**Model loading:**
```python
model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",  # BF16 base, NOT AWQ
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
```

**LoRA configuration (53M trainable params = 0.17% of 31B):**
```python
lora_config = LoraConfig(
    r=64, lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
)
```

**Training arguments:**
```python
TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # effective = 1 × 8 × n_gpus
    learning_rate=2e-4,             # higher than full FT (1e-5) — standard for LoRA
    bf16=True,
    deepspeed="ds_config_zero2.json",
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    warmup_ratio=0.03,
)
```

**Post-training pipeline:**
1. Merge adapter: `model.merge_and_unload()` → BF16 merged checkpoint (~61 GB)
2. AWQ quantize using `autoawq` with 100-500 FAR-domain calibration examples
3. Deploy to vLLM: `vllm serve ./qwen3vl-30b-awq --quantization awq`

**Expected training time on 2× H200 141GB:** ~4-8 hours for 3 epochs on ~5K-20K examples (est. 7-12 sec/step vs 40-60 sec/step on 8× A100 80GB).

---



## 9. Synthesis: The Cross-Cutting Risks for a FAR Institutional-Expert Model

Several findings interact in ways that compound risk rather than staying independent — and for this specific project, they interact with the security/classification concerns already established in the earlier military-deployment research.

1. **The access/scarcity double bind.** The most authoritative institutional knowledge (the full Discipline Générale Militaire text, internal Chief-of-Staff directives, actual case-handling procedure) is precisely the content that is hardest to legally and practically acquire, while what remains freely available (published dahirs, high-level org charts, pension formulas) is the shallower, more general layer — meaning open, easily-acquired data does not escape the underlying institutional-depth gap, and the highest-value training signal is also the highest-risk to handle from a security standpoint (Section 4 of the earlier governance research; see the companion report on FAR deployment architecture for the full classification-tiering framework). The multi-agent LLM-KE extraction pipeline (Section 3) mitigates this by extracting structured triples from whatever documents are accessible, but cannot create signal where no source text exists.
2. **Structured-to-text conversion loss compounds with tokenization loss, but vocabulary expansion is a validated fix.** Verbalizing FAR's organizational hierarchy and legal cross-reference graph into flat training text loses meaningful structural relationships even with strong LLM-assisted narration, and whatever survives that step then gets fragmented again by generic tokenizers that don't natively represent article/dahir numbering as stable units. Vocabulary expansion (+10K-15K FAR-specific tokens via continued BPE + embedding averaging init + 10K CPT steps) is now research-validated (EACL 2026, NeurIPS 2025) to reduce sequence length by 20-30% and improve token-adoption rates to ~98%, directly addressing the tokenization half of this compound loss. The structured half still requires a KG+RAG architecture.
3. **The hallucination paradox undermines the core motivation for fine-tuning.** The mechanism that makes fine-tuning valuable (injecting specialized, under-represented institutional knowledge) is the same mechanism shown to linearly increase hallucination once that knowledge is learned — and this is precisely the knowledge (exact article numbers, exact pension formulas, precise disciplinary procedure) where a hallucinated answer carries direct real-world consequences for a service member. This suggests SFT-heavy approaches to teaching FAR-specific regulatory detail are likely self-defeating without RAG grounding layered on top, mirroring the legal-domain finding that even fine-tuned "customized legal AI systems" still hallucinate at 17-33%. The target language mix (55% French, 30% Darija, 10% MSA, 5% code-switched) adds further complexity: Darija's unstandardized orthography compounds hallucination risk if the model learns to generate plausible-sounding Darija that is factually wrong.
4. **Fine-tuned domain models don't reliably self-audit better than general models.** Since general-purpose models with strong reasoning have matched or beaten domain-fine-tuned models at detecting their own field's hallucinations in comparable domains, the more robust architecture is likely retrieval-plus-reasoning-plus-verification rather than deep SFT alone — consistent with the pattern seen in the strongest legal-domain case studies (domain-partitioned hybrid RAG+KG, commercial RAG legal tools), which pair light fine-tuning or reranking with retrieval rather than relying on fine-tuning in isolation.
5. **Evaluation itself is unsolved, and uniquely so for a single non-public institution.** There is no existing benchmark that measures FAR-specific institutional competence, no general Moroccan-legal benchmark to fall back on, ground truth for multi-hop cross-legal-instrument reasoning is itself contested even where documentation exists, and LLM-judge evaluation carries its own domain-gap and self-preference bias problems — meaning any credible project needs a custom, expert-reviewed evaluation set built with genuine FAR administrative/legal expertise, and needs to accept that this expert annotation will be both costly and, per comparable domains, only moderately consistent even among genuine experts. Existing military-domain benchmarks (ARMOR 2025, WARBENCH, MEBL 2026) offer partial coverage of doctrinal/IHL scenarios but zero coverage of FAR-specific or Moroccan-legal content.
6. **MoE architecture imposes hard training constraints that are non-negotiable.** Qwen3-VL-30B-A3B is a Mixture-of-Experts model (128 experts, 8 active), which means: DeepSpeed ZeRO-3 is incompatible with LoRA on MoE (ZeRO-2 only), the router must not be fine-tuned (Unsloth disables this by default), and LoRA targets both attention and FFN projections across the 8 active experts per token (~53M params, 0.17% of total). The training framework must be either Unsloth (recommended, with native Qwen3 FastModel support) or TRL with manual ZeRO-2 config. Axolotl is a secondary option; LitGPT does not support MoE at all.

---

## Recommended Approach Given These Findings

Given the balance of evidence, a pragmatic pipeline for a FAR institutional-expert model on **2× H200 141GB (PCIe, PP=2)** with **Qwen3-VL-30B-A3B-Instruct** would combine:

**Phase 1 — Base deployment.** Deploy the AWQ-quantized model immediately on the inference stack (vLLM + FP8 KV cache + 250-user swap at 64K context). Establish baseline evaluation on custom FAR eval, MMLU, and ARMOR-adapted scenarios before any training begins. This gives a measurable starting point and a working RAG-grounded system (Qdrant) that the fine-tuned model will later augment, not replace.

**Phase 2 — Data pipeline.** Run multi-agent LLM-KE extraction across ~125 GB of documents, building a FAR knowledge wiki in Qdrant. Generate SFT pairs at the target language mix: 55% French, 30% Darija (explicitly labeled, not treated as Arabic), 10% MSA, 5% code-switched. Review 500 seed examples by hand before scaling to 20K+ synthetic.

**Phase 3 — Vocabulary expansion.** Train continued BPE on the FAR corpus to create ~10K-15K new tokens (dahir numbers, acronyms, French/Arabic/Darija terms appearing >100 times). Initialize embeddings via averaging of constituent sub-token embeddings. Run 10K CPT steps with 25% general replay. Verify cross-entropy loss per language before proceeding.

**Phase 4 — BF16 LoRA training via Unsloth** (DeepSpeed ZeRO-2, router frozen, ~53M trainable params). Train for 3 epochs on ~5K-50K examples with 75/25 FAR/general data mixing. Use the published recipe: r=64, α=128, lr=2e-4, GA=8, gradient checkpointing, bf16, cosine decay with 3% warmup. Expected time: ~4-40 hours depending on dataset size. Optionally apply SLoRA's post-hoc filter after training to recover ~29% of forgotten general capability at no extra compute cost ([ACL 2026](https://aclanthology.org/2026.acl-long.513/)).

**Phase 5 — Merge → AWQ quantize → evaluate.** Merge LoRA adapter into BF16 base, quantize to AWQ with 100-500 FAR-domain calibration examples, benchmark on custom FAR eval (200+ questions across DGM recall, multi-hop cross-refs, French/Darija fluency, MMLU retention, ARMOR-style doctrinal). Verify that AWQ quantization did not disproportionately degrade citation accuracy (the ~85.9% MedGemma finding is a caution here — test directly).

**Phase 6 — DPO alignment (optional).** If evaluation shows style/register issues (e.g., model answers in formal French when the user wrote in Darija), run KL-anchored DPO with β sweep {0.1, 0.3, 0.5}, measuring forgetting on general eval before selecting.

---

## Recommended Architecture for Best Performance

```
                    ┌─────────────────────────────┐
                    │   Qwen3-VL-30B-A3B-Instruct  │
                    │   (BF16 base, ~61 GB)        │
                    └────────┬────────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │        LoRA Adapter          │
              │   r=64, α=128               │
              │   target: q,k,v,o,gate,up,down│
              │   53M trainable params        │
              │   ~200-400 MB checkpoint      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Vocabulary Extension     │
              │   +10K-30K FAR tokens        │
              │   + Continued BPE merges     │
              │   + Embedding averaging init │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     RAG Layer (Qdrant)       │
              │   - Article-level indexing   │
              │   - Domain-tuned embeddings  │
              │   - Multi-hop retrieval     │
              │   - Source citation always   │
              └──────────────┬──────────────┘
                             │
              ┌──────────────┴──────────────┐
              │     Guardrails               │
              │   - ARMOR doctrinal check    │
              │   - Hallucination detection  │
              │   - Classification-tier      │
              │     access gating           │
              └─────────────────────────────┘
```

**Why this architecture (research-grounded):**

- **LoRA over full fine-tune:** "LLMs mostly acquire factual knowledge during pretraining; fine-tuning teaches them to use existing knowledge more efficiently, not to reliably absorb new facts" (Gekhman et al., 2024, [arXiv 2405.05904](https://arxiv.org/abs/2405.05904)). LoRA's 53M params (0.17% of 31B) are enough for register adaptation; deeper SFT risks the hallucination paradox without proportional benefit.
- **RAG always on:** Even fine-tuned legal AI hallucinates at 17-33% (Stanford 2025, [dho.stanford.edu](https://dho.stanford.edu/wp-content/uploads/Legal_RAG_Hallucinations.pdf)). RAG + source citation is the only path below that. The Qdrant vector store (already in the inference stack) grounds every answer in retrievable dahir/décret text.
- **Vocabulary expansion:** Shortens FAR sequences by 20%+, speeds inference by 20-30% (NeurIPS 2025, Herold et al., [arXiv 2509.26124](https://arxiv.org/abs/2509.26124)), and reduces hallucination on article numbers by making them atomic tokens rather than fragmented sub-tokens.
- **CPT only on embeddings (10K steps):** Full CPT on 125 GB of FAR text is overkill and risks 41% forgetting on a 30B model (from the scaling-law finding in Section 1, [arXiv 2406.04836](https://arxiv.org/html/2406.04836v1)). Vocabulary adaptation via continued BPE + embedding averaging init + 10K CPT with 25% replay achieves ~98% token adoption without degrading general capability.
- **Guardrails as a separate layer:** Hallucination detection and classification-tier access gating are non-negotiable for a system answering career-critical HR and disciplinary questions. These are architectural separations, not training objectives — they cannot be effectively learned through SFT alone.

The legal-domain finding that even fine-tuned, commercially deployed legal AI systems still hallucinate at 17-33% is the single most important result to keep in mind — the goal is not maximal fine-tuning depth but the minimum intervention that reliably improves FAR-specific institutional competence without degrading general reasoning, multilingual fluency, or — uniquely important in this domain relative to biology or medicine — without increasing the risk of leaking sensitive internal content through model weights.
