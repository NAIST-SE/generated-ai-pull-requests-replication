import unittest

from audit_utils import (
    JOIN_DELIMITER,
    artifact_type_from_url,
    parse_aggregate,
    parse_issue_numbers,
    prompt_block,
    reference_evidence,
    sha256_text,
)


class AuditUtilsTests(unittest.TestCase):
    def test_issue_numbers_preserve_order(self):
        self.assertEqual(parse_issue_numbers("2955, 3001"), [2955, 3001])
        with self.assertRaises(ValueError):
            parse_issue_numbers("2955,2955")

    def test_aggregate_round_trip(self):
        value = JOIN_DELIMITER.join(
            [
                "issue #2955: First body\nwith Markdown",
                "issue #3001: Second body\r\nwith text",
            ]
        )
        self.assertEqual(
            parse_aggregate(value, [2955, 3001], "issue_bodies"),
            ["First body\nwith Markdown", "Second body\r\nwith text"],
        )

    def test_url_type_requires_repository_and_number(self):
        self.assertEqual(
            artifact_type_from_url(
                "https://github.com/JabRef/jabref/issues/2955",
                "JabRef/jabref",
                2955,
            ),
            "issue",
        )
        self.assertEqual(
            artifact_type_from_url(
                "https://github.com/JabRef/jabref/pull/2955",
                "JabRef/jabref",
                2955,
            ),
            "pull_request",
        )
        self.assertEqual(
            artifact_type_from_url(
                "https://github.com/other/repo/issues/2955",
                "JabRef/jabref",
                2955,
            ),
            "unknown",
        )

    def test_reference_evidence(self):
        body = "This is a follow-up on 7ef2b3a and should finally fix #2955."
        evidence = reference_evidence(body, "JabRef/jabref", 2955)
        self.assertTrue(evidence["reference_found"])
        self.assertEqual(evidence["reference_evidence_type"], "closing_keyword")

    def test_historical_repository_alias(self):
        body = "See https://github.com/koppor/jabref/issues/244."
        without_alias = reference_evidence(body, "JabRef/jabref", 244)
        with_alias = reference_evidence(
            body,
            "JabRef/jabref",
            244,
            ("koppor/jabref",),
        )
        self.assertFalse(without_alias["reference_found"])
        self.assertTrue(with_alias["reference_found"])
        self.assertEqual(
            with_alias["reference_evidence_type"],
            "explicit_issue_url",
        )

    def test_prompt_block_and_hash(self):
        issues = [
            {"stored_title": "One", "stored_body": "Body one"},
            {"stored_title": "Two", "stored_body": "Body two"},
        ]
        block = prompt_block(issues)
        self.assertEqual(
            block,
            "Title: One\nBody: Body one\n---\nTitle: Two\nBody: Body two",
        )
        self.assertEqual(len(sha256_text(block)), 64)


if __name__ == "__main__":
    unittest.main()
