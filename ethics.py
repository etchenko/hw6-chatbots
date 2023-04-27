"""
Please answer the following ethics and reflection questions. 

We are expecting at least three complete sentences for each question for full credit. 

Each question is worth 2.5 points. 
"""

######################################################################################
"""
QUESTION 1 - Anthropomorphizing

Is there potential for users of your chatbot possibly anthropomorphize 
(attribute human characteristics to an object) it? 
What are some possible ramifications of anthropomorphizing chatbot systems? 
Can you think of any ways that chatbot designers could ensure that users can easily 
distinguish the chatbot responses from those of a human?
"""

Q1_your_answer = """

We don't think that users will anthropomorphize our chatbot, since it is incredibly simplistic and requires a rigid interaction
structure in order to function. The chatbot is unable to interact with the user in any way besides recommending movies, which makes it
harder to view it as human. However, the addition of human-like ticks (i.e. 'oh!', 'thanks!') and the fact that the chatbot responds in
perfect English means it's not impossible. 

In general, anthropomorphization of chatbot systems can be dangerous, for many reasons. On the extreme level, this can take the form of
humans believing that the chatbot has a consciousness and can feel (such as what happend with LaMDA), which can cause many issues for both
the person who believes the chatbot has consciousness, and the creators of the chatbot itself. In less extreme and more common cases, there is
a potential for people to begin replacing their actual human contact with chatbot contact if the chatbot 'feels' human enough. This can cause issues
for human connections and relationships. 

There are many ways in which designers can make the chatbot distinguishable from humans. Here we list a few ideas:
1. Have that chatbot type out its response very quickly. When communicating with humans, we either wait a little and get an instantanous response (i.e. when
    we send a text), or very slwoly, since humans are slow typers. By having the bot type text at superhuman speeds, we can tell we are not communicating with a human.
2. Every once in a while, the chatbot should communicate to the human that it is a chatbot. ChatGPT does this well, since in some responses it will tell you that it is a
    chatbot and thus cannot respond to a given prompt. The chatbot could also have a built in mechanism to just remind the user that it is a chatbot.
3. Chatbot designers can make sure to remove any output that could convey emotion. They could make sure that all of the responses the chatbot gives are emotionless, which would
    make people less likely to anthropomorphize it.


The issue with advances in chatbots is that many developers have as their goal making the chatbots appear more and more human. This can be incredibly harmful, especially if the 
chatbots are released on the internet, and are used to spread false information and hate speech.
"""

######################################################################################

"""
QUESTION 2 - Privacy Leaks

One of the potential harms for chatbots is collecting and then subsequently leaking 
(advertly or inadvertently) private information. Does your chatbot have risk of doing so? 
Can you think of ways that designers of the chatbot can help to mitigate this risk? 
"""

Q2_your_answer = """

Our chatbot does not have a risk of doing so. All of the information that the chatbot initially starts with is anonymized,
so there is no chance in that sense. The chatbot is also a local machine, and thus it does not pose the risk of leaking the data
 onto the internet. Finally, our chatbot will stop after it recommends movies to you, and all of the data regarding your likes will be erased,
so there is nothing left after you finish interacting with our chatbot. 

In order to help mitigate the risk of leaking data, here are a few ideas:
1. Have the chatbot run locally, thus meaning no data will be reported over the internet.
2. Delete all person information after each conversation, or if we want that information to persist, save that information locally in a cache rather than on the cloud.
3. Allow the users to decide how much of the conversations is kept and how much is erased, as well as where information is stored.
4. Encrypt all personal information that the chatbot receives.

"""

######################################################################################

"""
QUESTION 3 - Effects on Labor

Advances in dialogue systems, and the language technologies based on them, could lead to the automation of 
tasks that are currently done by paid human workers, such as responding to customer-service queries, 
translating documents or writing computer code. These could displace workers and lead to widespread unemployment. 
What do you think different stakeholders -- elected government officials, employees at technology companies, 
citizens -- should do in anticipation of these risks and/or in response to these real-world harms? 
"""

Q3_your_answer = """

Automation, regardless of whether it occurs in dialogue systems or in other areas of industry, should first and foremost be used as a way to ease human labor
and help people live their lives more easily. If we arrived at a point where all tasks necessary for survival and global functioning were automated, we would hope
that all human would be equally able to enjoy their lives. The current capitalist economic system however is incompatible with this type of ideology and the future of automation.
The motivation behind automation under capitalism is not to ease human labor, but rather to increase profits for the small capitalist owner class, while the workers are left holding the 
short end of the stick. Since the 1970s, automation and technology has allowed human productivity to increase dramatically, putting more and more money into the pockets of the wealthy few, 
while the wages of the working class have remained relatively stagnant. This means that rather than look forward to automation, workers actively fear it, since automation means unemployment
for the replaced workers. The simplest and most basic way to combat this to be considered by elected government officials is the introduction of a Universal Basic Income (UBI). As automation starts to 
take over more and more jobs, a UBI would allow those whose jobs are taken over by automation to still survive and live happy lives. However, UBI is just a temporary way to address the issue, and will 
not be able to solve the issues in a manner that will allow everyone to live well, because of the complexities behind worker motivation under UBI, calculating the right amount of UBI to still mantain a 
functioning society, and so on. A slightly better solution is for elected government officials to force a restructuring around the way workers interact with the business they work for. By forcing business 
to make the workers equal stakeholders in the business (the complexities of creating a truly socialist and worker - owned business is still a problem people are working on), workers are more incentivized to 
achieve automation, since their livelihoods will not be taken away if their jobs are automated. However, this will still not fully solve the issue, as businesses still operate with a 'profit first' mindset, which ultimately 
benefits nobody but the stakeholders in the short term. Ultimately, as automation becomes more and more prevalent, we as a society must shift further and further towards a socialist existence, one where a small minority 
of people do not hold a majority of the wealth, and one where everyone in the society equally benefts from the automation of a given industry, not just those in the business. Capitalism is incompatible with the longterm 
wellbeing of humans on the planet, and thus we must do everything in our power to dismantle the broken system that we live in and build something that won't kill the planet in 50 years. Thus, the only REAL longterm solution, 
and the best way for us to address the issues of automation, is to work towards building a socialist/communist society, one where automation is not an issue, but rather a blessing.

"""

"""
QUESTION 4 - Refelection 

You just built a frame-based dialogue system using a combination of rule-based and machine learning 
approaches. Congratulations! What are the advantages and disadvantages of this paradigm 
compared to an end-to-end deep learning approach, e.g. ChatGPT? 
"""

Q4_your_answer = """

There are a few advantages to this type of system over ChatGPT. Since this system is intended to do only one specific task, we only need to train models in order to do the task,
which means that we end up using a lot less compute power to train and run the model than is necessary for the end-to-end deep learning approaches. We are also able to update the rules 
as we see fit, and will not have to retrain the entire model when making updates. We can also incrementally add modules and more functionality if necessary, and remove functionality if it does
not become necessary. We can use the tools of vectorization that are already available rather than having to do so ourselves. However, unlike an end-to-end deep learning approach, our model is quite 
rigid, and thus may not work quite well in a lot of edge cases. As the functionality becomes more complicated, the rules will become more complex and more difficult to manage, while end-to-end systems
can be retrained.

"""