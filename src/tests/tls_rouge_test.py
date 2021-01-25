#%%
import datetime
import pprint

from tilse.data import timelines
from tilse.evaluation import rouge
from others.utils import cal_rouge_tls


def test_division_zero():
    (
        sent_str_combination,
        sent_date_combination,
        abstract_str_list,
        abstract_date_list,
    ) = (
        [
            "  Note for non  UK readers  a quick Google search can probably explain why he didn  t touch the pie "
        ],
        ["2016-01-18T11:27:05Z"],
        [
            "Is inducted into the Rock and Roll Hall of Fame ",
            "While on tour  visits a hospital in Hamburg  Germany  for treatment of pain in his arm ",
            " Lazarus   written by Bowie and Edna Walsh  opens off  Broadway at the New York City Workshop ",
            "The album  Blackstar  is released ",
            "Bowie dies at age 69 after an 18  month battle with cancer ",
            "Bowie receives four posthumous Grammy nominations for  Blackstar  ",
        ],
        [
            "1996-01-17 00:00:00",
            "2004-06-25 00:00:00",
            "2015-12-07 00:00:00",
            "2016-01-08 00:00:00",
            "2016-01-10 00:00:00",
            "2016-12-06 00:00:00",
        ],
    )
    rouge_1, rouge_2 = cal_rouge_tls(
        sent_str_combination,
        sent_date_combination,
        abstract_str_list,
        abstract_date_list,
    )
    print(rouge_1, rouge_2)


# test_division_zero()


def test_specific_case():
    (
        sent_str_combination,
        sent_date_combination,
        abstract_str_list,
        abstract_date_list,
    ) = (
        [
            " Bowie provided guest vocals to the band  s 2013 song Reflektor ",
            " But for those who have n  t  the hirsute one gives an interesting interview today where he says the infamous Mr Blobby could soon be starring in an animated series ",
        ],
        ["2016-01-20T03:10:10Z", "2005-11-22T15:30:57Z"],
        [
            "Is inducted into the Rock and Roll Hall of Fame ",
            "While on tour  visits a hospital in Hamburg  Germany  for treatment of pain in his arm ",
            " Lazarus   written by Bowie and Edna Walsh  opens off  Broadway at the New York City Workshop ",
            "The album  Blackstar  is released ",
            "Bowie dies at age 69 after an 18  month battle with cancer ",
            "Bowie receives four posthumous Grammy nominations for  Blackstar  ",
        ],
        [
            "1996-01-17 00:00:00",
            "2004-06-25 00:00:00",
            "2015-12-07 00:00:00",
            "2016-01-08 00:00:00",
            "2016-01-10 00:00:00",
            "2016-12-06 00:00:00",
        ],
    )
    rouge_1, rouge_2 = cal_rouge_tls(
        sent_str_combination,
        sent_date_combination,
        abstract_str_list,
        abstract_date_list,
    )
    print(rouge_1, rouge_2)
    (
        sent_str_combination,
        sent_date_combination,
        abstract_str_list,
        abstract_date_list,
    ) = (
        [" I haven  t even been charged "],
        ["2015-08-13T16:14:48Z"],
        [
            "A even classified military video is posted by WikiLeaks ",
            "The military announces it has charged Manning with violating army regulations by transferring classified information to a personal computer and adding unauthorized software to a classified computer system and of violating federal laws of governing the handling of classified information ",
            "WikiLeaks posts more than 90  000 classified documents relating to the Afghanistan war",
            "WikiLeaks publishes nearly 400  000 classified military documents from the Iraq War  providing a new picture of how many Iraqi civilians have been killed  the role that Iran has played in supporting Iraqi militants and many accounts of abuse by Iraq  s army and police ",
            "WikiLeaks begins publishing approximately 250  000 leaked State Department cables dating back to 1966 ",
            "The WikiLeaks website suffers an attack designed to make it unavailable to users ",
            "Amazon removes WikiLeaks from its servers ",
            "Nearly 800 classified US military documents obtained by WikiLeaks reveal details about the alleged terrorist activities of al Qaeda operatives captured and housed in Guantanamo Bay ",
            "WikiLeaks releases its archive of more than 250  000 unredacted US diplomatic cables ",
            "WikiLeaks announces that it is temporarily halting publication to  aggressively fundraise  ",
            "Manning  s Article 32 hearing  the military equivalent of a grand jury hearing that will determine whether enough evidence exists to merit a court  martial  begins ",
            "Manning is formally charged with aiding the enemy  wrongfully causing intelligence to be published on the Internet  transmitting national defense information and theft of public property or records ",
            "WikiLeaks begins releasing what it says are five million emails from the private intelligence company  Stratfor",
            "WikiLeaks begins publishing more than 2  4 million emails from Syrian politicians  government ministries and companies dating back to 2006 ",
            "Manning pleads guilty to some of the 22 charges against him  but not the most serious charge of aiding the enemy  which carries a life sentence ",
            "",
            "Manning is acquitted of aiding the enemy  but found guilty on 20 other counts  including violations of the Espionage Act ",
            "A military judge sentences Manning to 35 years in prison ",
            "Through a statement read on NBC  s Today show  Manning announces he wants to live life as a woman and wants to be known by his new name  Chelsea Manning ",
            "A Kansas judge grants Manning  s request for a formal name change from Bradley to Chelsea ",
            "WikiLeaks releases nearly 20  000 emails from Democratic National Committee staffers ",
            "More than 2  000 hacked emails from Clinton  s campaign chairman  John Podesta are published by WikiLeaks ",
            "During an interview on the Fox News Network  Assange says that Russia did not give WikiLeaks hacked emails ",
            "WikiLeaks tweets that Assange will agree to be extradited to the US if Obama grants clemency to Manning ",
            "Obama commutes Manning  s sentence  setting the stage for her to be released on May 17 ",
            "WikiLeaks publishes what they say are thousands of internal CIA documents  including alleged discussions of a covert hacking program and the development of spy software targeting cellphones  smart TVs and computer systems in cars ",
            "Authorities tell CNN that they are taking steps to seek the arrest of Assange ",
            "During a Senate hearing  FBI Director James Comey refers to WikiLeaks as  intelligence porn   declaring that the site  s disclosures are intended to damage the US rather than educate the public ",
            "Manning is released from prison ",
            "Harvard Kennedy School withdraws an invitation to Chelsea Manning to be a visiting fellow ",
            "Manning files to run as a Democratic candidate of Senate in Maryland  according to Federal Election Commission records ",
        ],
        [
            "2010-04-05 00:00:00",
            "2010-07-06 00:00:00",
            "2010-07-25 00:00:00",
            "2010-10-22 00:00:00",
            "2010-11-28 00:00:00",
            "2010-11-28 00:00:00",
            "2010-12-01 00:00:00",
            "2011-04-24 00:00:00",
            "2011-09-02 00:00:00",
            "2011-10-24 00:00:00",
            "2011-12-16 00:00:00",
            "2012-02-23 00:00:00",
            "2012-02-26 00:00:00",
            "2012-07-05 00:00:00",
            "2013-02-28 00:00:00",
            "2013-06-03 00:00:00",
            "2013-07-30 00:00:00",
            "2013-08-21 00:00:00",
            "2013-08-22 00:00:00",
            "2014-04-23 00:00:00",
            "2016-07-22 00:00:00",
            "2016-10-07 00:00:00",
            "2017-01-03 00:00:00",
            "2017-01-12 00:00:00",
            "2017-01-17 00:00:00",
            "2017-03-07 00:00:00",
            "2017-04-20 00:00:00",
            "2017-05-03 00:00:00",
            "2017-05-17 00:00:00",
            "2017-09-15 00:00:00",
            "2018-01-11 00:00:00",
        ],
    )
    rouge_1, rouge_2 = cal_rouge_tls(
        sent_str_combination,
        sent_date_combination,
        abstract_str_list,
        abstract_date_list,
    )
    print(rouge_1, rouge_2)


test_specific_case()


def test_offical_case():
    evaluator = rouge.TimelineRougeEvaluator(measures=["rouge_1", "rouge_2"])

    predicted_timeline = timelines.Timeline(
        {
            datetime.date(2010, 1, 1): ["Just a test .", "Another sentence ."],
            datetime.date(2010, 1, 3): ["Some more content .", "Even more !"],
        }
    )

    groundtruth = timelines.GroundTruth(
        [
            timelines.Timeline(
                {
                    datetime.date(2010, 1, 2): ["Just a test ."],
                    datetime.date(2010, 1, 4): ["This one does not match ."],
                }
            ),
            timelines.Timeline(
                {
                    datetime.date(2010, 1, 1): ["Just a test .", "Another sentence ."],
                    datetime.date(2010, 1, 3): ["Another timeline !"],
                }
            ),
        ]
    )

    pp = pprint.PrettyPrinter(indent=4)

    print("concat")
    pp.pprint(evaluator.evaluate_concat(predicted_timeline, groundtruth))

    print("align, date-content costs")
    pp.pprint(
        evaluator.evaluate_align_date_content_costs(predicted_timeline, groundtruth)
    )

    print("evaluate all")
    pp.pprint(evaluator.evaluate_all(predicted_timeline, groundtruth))
    assert 1


# test_offical_case()
# %%
