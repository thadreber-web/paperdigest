# PaperDigest — how I’d explain it

PaperDigest came from me trying to read newer AI papers and realizing that just having the PDF was not very useful. I could get the general idea, but then I would hit an equation, a figure, or some part of the method that assumed I already knew more than I did. Even when a paper had code, that did not necessarily make it easier to understand what the authors actually claimed or how I would start reproducing the work.

So I built a tool that takes a paper and gives me two things.

The first is a digest: structured Obsidian notes that break the paper down into sections, explain the concepts in plain English, keep the formulas, and include diagrams and figure explanations. The goal was not to replace the paper with a shallow summary. It was to make a hard paper possible to work through without constantly losing the thread.

The second is a scaffold. If I decide a paper is worth trying, PaperDigest can make a starting project around it: a package layout, configs, experiment notes, and the pieces that still need real implementation. I kept the digest as the default because understanding the paper comes before generating a bunch of code. The scaffold is there to shorten the distance between “this is interesting” and “I have a sensible place to start.”

I made it local-first on purpose. I have local models running, and I did not want the project to depend on sending papers to an API or making someone sign up for a service. It works with local OpenAI-compatible servers. Cloud models are an option, but they are not required.

A lot of the decisions came from seeing where the first versions broke in real use.

Smaller local models do not always return perfect JSON, so I changed the parser to handle a valid response even when the model keeps talking afterward. A 9B model also cut off a scaffold because the original token limit was too low. I raised it from 8K to 16K. I did not just set it to something huge, because long planning output can wander and waste time. Sixteen thousand was the practical middle ground from an actual failure.

I also learned that if the tool generates a project, I cannot trust it to invent every boring infrastructure piece correctly. At first, a scaffold could generate its own config parser. That meant every project could get a slightly different, untested parser. I changed that so PaperDigest ships a tested config loader instead. Same thing with imports and dependencies: the generated code needed cleanup so it would work as an installed project, not only as a pile of files from one model run.

The biggest rule I added was around experimental results. A model should not be allowed to look at a paper and make up benchmark numbers, rename metrics, or write claims as if they came from the authors. PaperDigest now carries results from the paper-reading step into the experiment document. If something is stated in the paper, it can say that. If it is an implementation choice or something unknown, it should say that instead.

I have also had to correct the tool based on actual output. In one run, the system described a figure as broken when it was not. The real problem was a Mermaid template artifact elsewhere in the output. That mattered because the right fix was not “the figure extraction is broken.” It was better template handling and better checking of the model’s figure descriptions.

The project is complete in the sense that it works end to end and has examples. It is not “finished forever.” The main remaining direction is adding support for more than arXiv without taking the easy route of converting every journal paper through a PDF pipeline that wrecks the formatting. I would rather support fewer sources well than claim broad support while making the paper worse on the way in.

That is really what PaperDigest is: a tool I built because I wanted a better way to go from a hard research paper to understanding it, studying it, and—when it is worth it—having an honest starting point for trying it myself.
