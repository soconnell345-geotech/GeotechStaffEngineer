"""The conventions the owner's two schemas are answered under, as DATA.

WHY THIS IS A FILE OF ITS OWN. The first full cluster run (ledger run 7, eight
reports) put the narrative reader at 64 % recall and 74 % precision, and the
per-question table said the losses were not comprehension. They were
CONVENTION: the reader wrote ``null`` where the hand writes ``0`` for a kind
of exploration the report plainly did not do; it answered the four "mention"
questions on whether the report *studied* a topic where the hand answers on
whether it *discusses* it; it filled ``earthHazardsExposed`` with seismic
shaking, which the hand never lists. Those are not things a better model gets
right. They are house rules, and a house rule belongs somewhere a person can
read it, disagree with it and change it in one place -- not inside a prompt
string.

So the rules live here, each with a STATUS. ``DRAFT`` means the lead wrote it
from the evidence of the scorecard and the owner has not yet confirmed it;
``CONFIRMED`` means the owner has. Nothing here is enforced in Python: the
block goes into the reader's prompt and the reader answers under it, so
changing a rule changes the answers and changes nothing else.

FIELD NAMES ARE VERBATIM, always, in both schemas. A rule keyed to a name the
schema does not have is a rule that will never fire, and :func:`unknown_fields`
exists so a test can say so.

THE OPEN QUESTION this file is the home for: the plan's section 7 item, the
owner's review of the two query schemas. Every ruling that review produces is
a line here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

__all__ = [
    "Convention", "CONVENTIONS", "FIELD_NOTES", "STATUSES",
    "glossary_block", "conventions_for", "unknown_fields",
]

#: What a rule's status may be. A DRAFT rule is in force and marked as
#: unconfirmed; there is no third state, because a rule nobody applies is not
#: a rule.
STATUSES: Tuple[str, ...] = ("DRAFT", "CONFIRMED")


@dataclass(frozen=True)
class Convention:
    """One house rule about how a question is answered.

    ``fields`` are the owner's field names it governs, verbatim, and may be
    empty for a rule that governs the whole schema. ``why`` is for the person
    reading the file, never for the prompt.
    """

    rule: str
    fields: Tuple[str, ...] = ()
    status: str = "DRAFT"
    why: str = ""

    def line(self) -> str:
        """The rule as the prompt prints it."""
        where = (" [" + ", ".join(self.fields) + "]") if self.fields else ""
        mark = " (DRAFT)" if self.status == "DRAFT" else ""
        return f"{self.rule}{where}{mark}"


#: The rules, in the order the prompt prints them. The six marked DRAFT are
#: the lead's reading of the per-question table of the eight-report run
#: (2026-09-18) and the owner is to confirm or overrule each one.
CONVENTIONS: Tuple[Convention, ...] = (
    Convention(
        rule="A count of explorations the report says it did NOT do is 0, "
             "not null. A report that describes four borings and says no "
             "test pits were dug, or that describes a programme of borings "
             "and never mentions a test pit or a cone at all, answers "
             "testPitCount 0 and cptCount 0. Null is for a report that "
             "leaves you unable to tell.",
        fields=("boringCount", "testPitCount", "cptCount",
                "previousInvestigationCount", "structureCount"),
        status="DRAFT",
        why="run 7: cptCount missed on 7 of 8 reports and testPitCount on "
            "5 -- the hand writes 0, the reader wrote null"),
    Convention(
        rule="null means THE REPORT DOES NOT SAY. It never means zero, "
             "never means not applicable, and is never written as a word: "
             "not the string \"none\", \"N/A\", \"unknown\" or an empty "
             "string.",
        fields=(),
        status="DRAFT",
        why="the distinction the whole scorecard rests on: a report that "
            "did not say has to be distinguishable from a reader that did "
            "not read"),
    Convention(
        rule="The four \"mention\" questions are about whether the report "
             "DISCUSSES the topic anywhere in it, not about whether the "
             "work was done, not about whether the finding was positive. "
             "A report that says in one sentence that corrosion testing "
             "was not performed HAS discussed soil corrosion: the answer "
             "is yes, and the reason says what it says. Answer no only "
             "where the report is silent on the topic throughout.",
        fields=("geophysicalTestingMention", "soilCorrosion",
                "siteResponseMention", "hazardAnalysisMention"),
        status="DRAFT",
        why="run 7: siteResponseMention WRONG on 7 of 8, soilCorrosion and "
            "hazardAnalysisMention on 3 each -- the reader was answering a "
            "different question from the one the hand answers"),
    Convention(
        rule="postName is the CITY of the diplomatic post, in the city's "
             "own name and nothing else: not the country, not the compound, "
             "not the project. It stays null whenever outsideProject is "
             "yes.",
        fields=("postName",),
        status="DRAFT",
        why="run 7: postName missed on 3 of 4 reports that have one"),
    Convention(
        rule="primeAe is the ARCHITECT-ENGINEER OF RECORD for the project "
             "-- the design firm the geotechnical engineer reports to or "
             "prepared the report for. It is not the geotechnical firm "
             "itself, which is geotechnicalEngineerFirm, and not the "
             "construction contractor, which is primeContractor.",
        fields=("primeAe", "primeContractor", "geotechnicalEngineerFirm"),
        status="DRAFT",
        why="run 7: primeAe missed on 3 of 5; the three names sit together "
            "on the letter and the cover and were being confused"),
    Convention(
        rule="earthHazardsExposed uses the listed phrases and no others, "
             "and NEVER includes seismic shaking. Ground shaking is on "
             "every site in a seismic region and listing it says nothing; "
             "this field is for what the report says THIS site is exposed "
             "to beyond it. An empty answer is right for a report that "
             "names no hazard.",
        fields=("earthHazardsExposed",),
        status="DRAFT",
        why="run 7: 3 wrong and 5 invented, almost all of them seismic "
            "shaking added to a list the hand left without it"),
)

#: A one-line gloss for a field whose name does not say what it wants. Keyed
#: by the owner's field name, verbatim. These are notes, not rules: they
#: carry no status because nothing about them is in dispute.
FIELD_NOTES: Dict[str, str] = {
    "documentType": "what kind of document this is, as a whole",
    "projectNumber": "the number the report itself prints for the project, "
                     "whoever assigned it",
    "projectPhase": "the stage of the project this report was written for",
    "outsideProject": "yes when this report was written for a client other "
                      "than the owner of the estate the library is about",
    "structureList": "the structures the report is about, one short name "
                     "each, as the report names them",
    "boringDictionary": "the identifiers of the borings, exactly as printed "
                        "-- B-1, SB-04, LB-05 -- and nothing else",
    "testPitDictionary": "the identifiers of the test pits, exactly as "
                         "printed",
    "recommendedFoundations": "one item per foundation type recommended, in "
                              "the report's own words",
    "bearingCapacity": "the report's own sentences about allowable or net "
                       "bearing pressure, one per recommendation",
    "strata": "the soil and rock profile in one string, as the report words "
              "it",
    "siteClass": "the seismic site class letter the report assigns",
    "asceSevenVersion": "the edition of ASCE 7 the report used, as it prints "
                        "it",
    "seismicCodeUsed": "the code or standard the seismic parameters were "
                       "taken from, as printed",
    "liquefactionPotential": "the report's own verdict on this site",
    "reportDate": "the date on the report itself, not the date of the field "
                  "work",
    "tableCount": "how many tables the report contains",
    "figureCount": "how many figures the report contains",
}


def conventions_for(field_name: str) -> List[Convention]:
    """Every rule that governs one field, general rules included."""
    return [row for row in CONVENTIONS
            if not row.fields or field_name in row.fields]


def unknown_fields(known: Sequence[str]) -> List[str]:
    """Field names used here that the two schemas do not have.

    A rule or a note keyed to a name the schema does not carry is dead text
    in the prompt, and a test calls this with
    ``GENERAL_FIELDS + NATURAL_HAZARD_FIELDS`` so it cannot stay dead.
    """
    have = set(known)
    used = set(FIELD_NOTES)
    for row in CONVENTIONS:
        used.update(row.fields)
    return sorted(used - have)


def glossary_block() -> str:
    """The conventions and the field notes, as the prompt prints them."""
    lines = [
        "THE HOUSE CONVENTIONS. These are how THIS library answers these "
        "questions, and they override any other reading. Rules marked "
        "(DRAFT) are being confirmed and are in force meanwhile.",
    ]
    for number, row in enumerate(CONVENTIONS, start=1):
        lines.append(f"  {number}. {row.line()}")
    lines += ["", "WHAT EACH FIELD MEANS, where the name does not say"]
    for name, note in FIELD_NOTES.items():
        lines.append(f"  {name}: {note}")
    return "\n".join(lines)
