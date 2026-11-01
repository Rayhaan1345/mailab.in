# Step 2.5: shared snippets, and what changed on the pages

Step 2.5 moved text that repeats across pages into `lib/snippets/`. A page writes `{{snippet:key}}` and the build puts the snippet text there. The README says how snippets work and lists the keys. This file is for review. It lists the wording that changed on the pages, because a snippet has one wording and some pages said the same thing in different words.

Page names are chapter codes: A1 is `a1-what-is-a-model`, D6 is `d6-long-context`, PA is `pa-practice`.

## What was compared

The built `index.html` of all 47 pages was compared with the copy from before the step (the state after step 2.4). Both were read with an HTML parser and compared as a stream of tags, attributes (in any order) and text (white space collapsed). The generated comment and the `?v=` stamps were ignored. The script is `lib/compare_pages.py`. With `--changes` it checks this file against the pages.

- 28 pages are byte for byte the same as before. A snippet that has the text a page already had changes no byte.
- 19 pages differ: 16 chapters and 3 practice sets. All differences are in the entries below. The entries were applied to the old stream of the page, and the result was equal to the new stream.
- No fact was changed. Where a page gains words, they state a fact the book already states elsewhere (see "Added words" at the end).

The entries use the form `was` and `now`. They are HTML as the parser sees it: attributes sorted, entities turned into characters.

## Entries

### a1-what-is-a-model

Snippet `sharks-made-up`.

- was: `All the sharks here are made up: records generated from published length-weight relationships, not field data.`
  now: `The sharks are made up: records generated from published length-weight relationships, not field data.`

### a4-grading-models

Snippets `sharks-made-up`, `pima-data`, `diabetes-lesson`, `diabetes-subgroups`, `smith-1988-adap`.

- was: `The records are made-up, generated from published length-weight relationships; they are not field data.`
  now: `The sharks are made up: records generated from published length-weight relationships, not field data.`
- was: `it has 768 women of Pima heritage, aged 21 or older, measured in the United States (Smith et al., 1988).`
  now: `it has 768 women of Pima heritage, measured in the United States, all aged 21 or older (Smith et al., 1988).`
- was: `A model trained on one population may not suit another, a lesson I drew in my <a href="https://rayhaan1345.github.io/projects/diabetes-screening/">diabetes screening project</a>.`
  now: `The lesson I drew in my <a href="https://rayhaan1345.github.io/projects/diabetes-screening/">diabetes screening project</a> is that a model trained on one population may not suit another.`
- was: `where I compared results across demographic subgroups to look for bias.`
  now: `where I checked results across demographic subgroups for bias.`
- was: `<i>Proc Annu Symp Comput Appl Med Care</i>, 261–265.`
  now: `<i>Proceedings of the Annual Symposium on Computer Application in Medical Care</i>, 261–265.`

### b7-continual-learning

Snippets `ttt-method`, `ttt-no-pause`, `ttt-port`, `ttt-motivation`, `ttt-links`. The bridge opens with the method now. The sentence about the port follows it.

- was: `My work at MBZUAI ports In-Place Test-Time Training, a method from ByteDance Seed, from text models to a Qwen vision-language model. In-Place TTT treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input, so generation never pauses.`
  now: `In-Place Test-Time Training, a method from ByteDance Seed (Feng and others), treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. Generation never pauses: an update happens in the step that completes a chunk. At MBZUAI, in Prof. Salman Khan's group, I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`
- was: `My motivation is the question of this chapter: recognising a shark when the water, the light and the animals change between dives.`
  now: `My motivation is the question of this chapter: recognising a shark when the water, light and animals change between dives.`
- was: `See the <a href="https://rayhaan1345.github.io/projects/in-place-ttt/">project page</a> and its <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">primer</a>, and <a href="../d6-long-context/#ttt">D6</a> for the long-context side.`
  now: `See the <a href="https://rayhaan1345.github.io/projects/in-place-ttt/">project page</a> and its <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">primer</a>. <a href="../d6-long-context/#ttt">D6</a> covers the long-context side.`

### c3-domain-shift

Snippets `ttt-method`, `ttt-no-pause`, `ttt-port`, `ttt-motivation`, `ttt-primer-ref`. The bridge keeps its own first sentence, without the place of work, which `ttt-port` now gives.

- was: `My work at MBZUAI, in Salman Khan's group, uses a loop of the same kind on a language model.`
  now: `My work uses a loop of the same kind on a language model.`
- was: `In-Place TTT, from ByteDance Seed, treats the down projection of selected MLP blocks of a language model as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input, and generation never pauses. I am porting the method from text models to a Qwen vision-language model so that it can adapt to an input after deployment.`
  now: `In-Place Test-Time Training, a method from ByteDance Seed (Feng and others), treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. Generation never pauses: an update happens in the step that completes a chunk. At MBZUAI, in Prof. Salman Khan's group, I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`
- was: `My motivation is the question of this chapter: recognising a shark when the water, the light and the animals change between dives.`
  now: `My motivation is the question of this chapter: recognising a shark when the water, light and animals change between dives.`
- was: `Khan, R. A primer on In-Place test-time training. <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">rayhaan1345.github.io/projects/in-place-ttt/primer</a>.`
  now: `Khan, R. A primer on In-Place Test-Time Training. <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">rayhaan1345.github.io/projects/in-place-ttt/primer/</a>.`

### d1-tokenisation

Snippet `gpt2-licence`.

- was: `GPT-2 has the MIT licence on its <a href="https://huggingface.co/gpt2">model card</a>, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`
  now: `The GPT-2 <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`

### d3-next-token

Snippet `gpt2-licence`.

- was: `Its <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`
  now: `The GPT-2 <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`

### d4-self-attention

Snippet `gpt2-licence`.

- was: `Its <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`
  now: `The GPT-2 <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`

### d5-transformer

Snippets `ttt-method`, `ttt-port`, `gpt2-licence`, `ttt-primer-ref`.

- was: `In-Place Test-Time Training, a method from ByteDance Seed, treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk while the model reads, with a target built from the input. I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`
  now: `In-Place Test-Time Training, a method from ByteDance Seed (Feng and others), treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. At MBZUAI, in Prof. Salman Khan's group, I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`
- was: `The model card gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`
  now: `The GPT-2 <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`
- was: `Khan, R. A primer on In-Place Test-Time Training. <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">rayhaan1345.github.io/projects/in-place-ttt/primer</a>.`
  now: `Khan, R. A primer on In-Place Test-Time Training. <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">rayhaan1345.github.io/projects/in-place-ttt/primer/</a>.`

### d6-long-context

Snippets `ttt-method`, `ttt-no-pause`, `ttt-port`, `gpt2-citation`, `gpt2-pdf-link`, `gpt2-licence`, `ttt-primer-ref`. The link to D5 stays, in a sentence of its own.

- was: `In-Place Test-Time Training, a method from ByteDance Seed, treats the down projection of selected MLP blocks (<a href="../d5-transformer/#block">D5</a>) as the fast weight.`
  now: `In-Place Test-Time Training, a method from ByteDance Seed (Feng and others), treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. The MLP blocks are those of <a href="../d5-transformer/#block">D5</a>.`
- was: `Generation never pauses: the update happens in the step that completes a chunk, and a chunk is read with the weights from before its own update.`
  now: `Generation never pauses: an update happens in the step that completes a chunk. A chunk is read with the weights from before its own update.`
- was: `I am porting the method to a Qwen vision-language model, so that it can adapt to an input after deployment.`
  now: `At MBZUAI, in Prof. Salman Khan's group, I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`
- was: `Radford, A. and others (2019). Language Models are Unsupervised Multitask Learners (GPT-2). MIT licence.`
  now: `Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI. <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">PDF</a>. The GPT-2 <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`
- was: `Khan, R. A Primer on In-Place Test-Time Training.`
  now: `Khan, R. A primer on In-Place Test-Time Training.`

### d7-assistants

Snippets `gpt2-citation`, `gpt2-pdf-link`, `gpt2-licence`.

- was: `Radford, A. and others (2019). Language Models are Unsupervised Multitask Learners (GPT-2). MIT licence.`
  now: `Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI. <a href="https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf">PDF</a>. The GPT-2 <a href="https://huggingface.co/gpt2">model card</a> gives the MIT licence, and the licence file in OpenAI's <a href="https://github.com/openai/gpt-2">repository</a> is a Modified MIT License.`

### e2-clip

Snippets `ttt-method`, `ttt-port`, `ttt-motivation`. The bridge now says what the method does, as the other bridges do.

- was: `In my MBZUAI project, I am porting In-Place Test-Time Training, a method from ByteDance Seed, from text models to a Qwen vision-language model. The aim is a model that adapts to an input after deployment.`
  now: `In-Place Test-Time Training, a method from ByteDance Seed (Feng and others), treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. At MBZUAI, in Prof. Salman Khan's group, I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`

### e4-vlms

Snippets `ttt-method`, `ttt-no-pause`, `ttt-port`, `ttt-motivation`, `ttt-links`. The attribution "Feng and others" moved from the middle of the bridge to the method sentence.

- was: `My project at MBZUAI, in Prof. Salman Khan's group, ports In-Place Test-Time Training from text models to a Qwen vision-language model, so that it can adapt to an input after deployment. In-Place TTT (Feng and others, from ByteDance Seed) treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. Generation does not pause.`
  now: `In-Place Test-Time Training, a method from ByteDance Seed (Feng and others), treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. Generation never pauses: an update happens in the step that completes a chunk. At MBZUAI, in Prof. Salman Khan's group, I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.`
- was: `See <a href="https://rayhaan1345.github.io/projects/in-place-ttt/">my project page</a> and <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">my step-by-step primer</a>.`
  now: `See the <a href="https://rayhaan1345.github.io/projects/in-place-ttt/">project page</a> and its <a href="https://rayhaan1345.github.io/projects/in-place-ttt/primer/">primer</a>.`

### f3-sharkcam

Snippets `ttt-motivation`, `socratic-skills`.

- was: `My motivation for the TTT work is the same question as SharkCam's: recognising a shark when the water, the light and the animals change between dives.`
  now: `My motivation for the TTT work is the same question as SharkCam's: recognising a shark when the water, light and animals change between dives.`
- was: `The tutor is a plugin made of two skills (socratic-tutor and quiz-me).`
  now: `The tutor is a plugin made of two skills, <code>socratic-tutor</code> and <code>quiz-me</code>.`

### pa-practice

Snippet `catch-table`. The two sentences became one, as in the chapters.

- was: `The shark questions use the made-up SharkCam catch table. It has 600 records generated from published length-weight relationships and is not field data (<code>primer/data/sharks/catch.csv</code>).`
  now: `The shark questions use the made-up SharkCam catch table: 600 records generated from published length-weight relationships, not field data (<code>primer/data/sharks/catch.csv</code>).`

### pb-practice

Snippet `catch-table`.

- was: `The shark questions use the made-up SharkCam catch table. It has 600 records generated from published length-weight relationships and is not field data (<code>primer/data/sharks/catch.csv</code>).`
  now: `The shark questions use the made-up SharkCam catch table: 600 records generated from published length-weight relationships, not field data (<code>primer/data/sharks/catch.csv</code>).`

### pf-practice

Snippet `pima-data`.

- was: `The Pima data is a published dataset of women of Pima heritage, measured in the United States, and not data from Mumbai.`
  now: `The Pima data is a published dataset of 768 women of Pima heritage, measured in the United States, and not data from Mumbai.`

### t1-python

Snippet `catch-table`.

- was: `The file used from here on is the SharkCam catch table: 600 made-up shark records generated from published length-weight relationships, not field data.`
  now: `The file used from here on is the made-up SharkCam catch table: 600 records generated from published length-weight relationships, not field data.`

### t3-algebra-functions

Snippets `sharks-made-up`, `catch-ref`. The reference entry loses the `code` tags and the note about the laws moves behind the shared text.

- was: `The records are made up from published length-weight laws and are not field data.`
  now: `The sharks are made up: records generated from published length-weight relationships, not field data.`
- was: `Mumbai AI Lab (2026). Made-up catch records generated from published length-weight relationships, <code>primer/data/sharks/catch.json</code> and <code>species.json</code>. The laws and their sources are listed in <code>species.json</code>. The records are not field data.`
  now: `Mumbai AI Lab (2026). Made-up catch records generated from published length-weight relationships, primer/data/sharks/catch.json and species.json. The records are not field data. The laws and their sources are listed in <code>species.json</code>.`

### t8-charts

Snippet `catch-table`.

- was: `All the data comes from the SharkCam catch table: 600 made-up shark records generated from published length-weight relationships, not field data.`
  now: `All the data comes from the made-up SharkCam catch table: 600 records generated from published length-weight relationships, not field data.`

## Added words

- `ttt-method` names "Feng and others" for In-Place TTT in B7, C3, D5, D6 and E2. E4 and the source lists of D6 and E4 already credit them.
- `ttt-port` gives the place of work ("At MBZUAI, in Prof. Salman Khan's group") to D5, D6 and E2, as B7, C3 and E4 had it. It is in PROJECTS.md.
- E2 gains the two sentences that say what the method does, from `ttt-method`.
- D6 and D7 list the GPT-2 paper in full, with its PDF and the licence statement, as D1 to D5 did.
- `pf-practice` gains the number 768 for the Pima women, which A4 states.

## What the step did not change

- Pages whose snippet has the words they already had are byte for byte the same as before. This covers the Pima phrase on eight pages (A1, A5, A6, B1, F1, PA, T1, T7), the project addresses, the TMLR status, the Hindi and Marathi note, the photo notes of E1 to E4, the MNIST licence and the practice-set boilerplate.
- The intro line of "Come back later" has 10 wordings on 40 pages. It is an instruction to the reader that varies with the chapter ("without scrolling up" on the toolkit chapters, "say the answer aloud" on some). It was left. If one wording is wanted, it is one new snippet and 40 tokens.
- The short forms "made-up records, not field data" in figure notes are under 12 words and were left.
- Papers that several chapters cite in their own style were left: He and others (2016), Vaswani and others (2017), Karpathy (2019), Sanderson, Alammar, Kingma and Ba. The entries differ in how many authors they list and in the notes after them, so unifying them is a choice about the bibliography style, which belongs to the owner.
