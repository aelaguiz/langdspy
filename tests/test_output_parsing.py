import pytest
import sys

import logging

logger = logging.getLogger("langdspy")
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
logger.addHandler(handler)

from langdspy.field_descriptors import InputField, OutputField, OutputFieldBool, InputFieldDictList, HintField, InputFieldList
from langdspy.prompt_strategies import PromptSignature, DefaultPromptStrategy
from langdspy.prompt_runners import PromptRunner
from langdspy.formatters import as_multiline


class TestOutputParsingPromptSignature(PromptSignature):
    ticket_summary = InputField(name="Ticket Summary", desc="Summary of the ticket we're trying to analyze.")
    buyer_issues_summary = OutputField(name="Buyer Issues Summary", desc="Summary of the issues this buyer is facing.")
    buyer_issue_category = OutputField(name="Buyer Issue Enum", desc="One of: ACCOUNT_DELETION, BOX_CONTENTS_CUSTOMIZATION, BRAZE_UNSUBSCRIBE, CANCEL_SUBSCRIPTION, CHANGE_ADDRESS, CHARGE_DISCREPANCY, CHECKOUT_ERROR, COUPON_QUESTION, CUSTOM_SHIPPING_REQUEST, MISROUTED_TICKET, DONATION_REQUEST, DUPLICATE_ORDER, FAILED_REFUND, GIFT_MESSAGE, GUEST_CHECKOUT_UNCLAIMED, INTERNATIONAL_CHARGES, MANAGE_PAYMENT_INFO, MULTIPLE_ACCOUNTS, NPS_CX_FEEDBACK, PASSWORD_RESET, PRE_PURCHASE_QUESTION, SELLER_FEEDBACK, SELLER_NUDGE_OUTREACH, SYSTEM_ERROR, WHERE_IS_MY_ORDER - The specific issue that the buyer is facing.")

def test_output_parsing_with_extra_fields():
    prompt_runner = PromptRunner(template_class=TestOutputParsingPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    input_data = {
        "ticket_summary": """Ticket ID: 338355
Status: Open
Subject: Bump box personalization
Priority: 2
Group: MP Buyer
Created At: 2024-03-17 20:42:40
Last Updated: 2024-03-17 20:42:41
Minutes spent waiting: 35 minutes
Requester: REQUESTER
Responder ID: None - Responder Name: None
Tags: None
Description: [Chatbot]:  Hi there, how can we help you today?   [user]:  Trying to order boxes and put in personaliZation and when I hit done nothing changes   [Chatbot]:  (No Intent Predicted)   [Chatbot]:  I'm sorry for the inconvenience. It seems like you're trying to personalize your order, but it's not updating. Please ensure that you select your desired variants (such as color or size preferences) before adding the product to cart. If this doesn't resolve the issue, there might be a technical glitch we need to look into further. * Ordering Multiple Boxes From The Marketplace   [Chatbot]:  Was I able to help you resolve your question?
(Yes, thank you!)
(No, I need more help.)   [user]:  (No, I need more help.)   [Chatbot]:  I apologize that I'm not able to provide you with a resolution through this chat.   In order to further assist you with this question our customer service team is available to help you. Please complete the below form to be and someone from our team will contact you shortly to resolve your concern.   [Chatbot]:  Thank you for chatting with us.   [user]:  Name: [Penny keefer] Email: [keefer.penny1960@gmail.com] Subject: [Bump box personalization] Question: [Trying to order boxes and put in personaliZation and when I hit done nothing changes  also need to know how long for delivery]   [Chatbot]:  (Uploading attachments)   [Chatbot]:  (Uploading attachments)
Messages:"""
    }
    
    output_data = """<Buyer Issues Summary>The buyer is trying to personalize their order by selecting variants like color or size, but after making their selections and hitting "done", the changes are not being reflected. They are also asking how long delivery will take.</Buyer Issues Summary>
<Buyer Issue Enum>BOX_CONTENTS_CUSTOMIZATION</Buyer Issue Enum>
Unfortunately, based on the provided input, I do not have enough context to determine how tickets like this have typically been handled in the past or provide relevant agent responses and resolutions. The input only contains marketing emails from a company called Little Poppy Co. promoting their products. Without any actual support ticket details or previous agent responses, I cannot provide a meaningful output for this particular request.
"""
    
    config = {"llm_type": "anthropic"}
    result = prompt_runner.template.parse_output_to_fields(output_data, config["llm_type"])
    
    assert result["buyer_issues_summary"] == "The buyer is trying to personalize their order by selecting variants like color or size, but after making their selections and hitting \"done\", the changes are not being reflected. They are also asking how long delivery will take."
    assert result["buyer_issue_category"] == "BOX_CONTENTS_CUSTOMIZATION"

def test_output_parsing_with_missing_fields():
    prompt_runner = PromptRunner(template_class=TestOutputParsingPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    input_data = {
        "ticket_summary": """Ticket ID: 338355
Status: Open
Subject: Bump box personalization
Priority: 2
Group: MP Buyer
Created At: 2024-03-17 20:42:40
Last Updated: 2024-03-17 20:42:41
Minutes spent waiting: 35 minutes
Requester: REQUESTER
Responder ID: None - Responder Name: None
Tags: None
Description: [Chatbot]:  Hi there, how can we help you today?   [user]:  Trying to order boxes and put in personaliZation and when I hit done nothing changes   [Chatbot]:  (No Intent Predicted)   [Chatbot]:  I'm sorry for the inconvenience. It seems like you're trying to personalize your order, but it's not updating. Please ensure that you select your desired variants (such as color or size preferences) before adding the product to cart. If this doesn't resolve the issue, there might be a technical glitch we need to look into further. * Ordering Multiple Boxes From The Marketplace   [Chatbot]:  Was I able to help you resolve your question?
(Yes, thank you!)
(No, I need more help.)   [user]:  (No, I need more help.)   [Chatbot]:  I apologize that I'm not able to provide you with a resolution through this chat.   In order to further assist you with this question our customer service team is available to help you. Please complete the below form to be and someone from our team will contact you shortly to resolve your concern.   [Chatbot]:  Thank you for chatting with us.   [user]:  Name: [Penny keefer] Email: [keefer.penny1960@gmail.com] Subject: [Bump box personalization] Question: [Trying to order boxes and put in personaliZation and when I hit done nothing changes  also need to know how long for delivery]   [Chatbot]:  (Uploading attachments)   [Chatbot]:  (Uploading attachments)
Messages:"""
    }
    
    output_data = """<Buyer Issues Summary>The buyer is trying to personalize their order by selecting variants like color or size, but after making their selections and hitting "done", the changes are not being reflected. They are also asking how long delivery will take.</Buyer Issues Summary>
Unfortunately, based on the provided input, I do not have enough context to determine how tickets like this have typically been handled in the past or provide relevant agent responses and resolutions. The input only contains marketing emails from a company called Little Poppy Co. promoting their products. Without any actual support ticket details or previous agent responses, I cannot provide a meaningful output for this particular request.
"""
    
    config = {"llm_type": "anthropic"}
    result = prompt_runner.template.parse_output_to_fields(output_data, config["llm_type"])
    
    assert result["buyer_issues_summary"] == "The buyer is trying to personalize their order by selecting variants like color or size, but after making their selections and hitting \"done\", the changes are not being reflected. They are also asking how long delivery will take."
    assert result.get("buyer_issue_category") is None

def test_repeated_input_output():
    output_data = """<input_fields>
<Ticket Summary>A summary of the ticket</Ticket Summary>
</input_fields>

<output_fields>
<Spam>One of: <choices>Yes, No</choices> - Is this ticket a spam or sales ticket</Spam>
</output_fields>

<examples>

<example>
<input>
<Ticket Summary>«Ticket ID: 2044
Freshdesk ID: 335398
Status: PENDING
Processing State: TRIAGED_READY
Subject: Horror collection
Priority: 2
Messages:»</Ticket Summary>
</input>
<output>
<Spam>No</Spam>
</output>
</example>

<example>
<input>
<Ticket Summary>«Ticket ID: 2504
Freshdesk ID: 334191
Status: PENDING
Processing State: TRIAGED_READY
Subject: Ch
Messages:»</Ticket Summary>
</input>
<output>
<Spam>No</Spam>
</output>
</example>
</examples>

<input>
<Ticket Summary>«Ticket ID: 2453
Freshdesk ID: 334312
Status: IN_PROGRESS
Processing State: TRIAGED_READY
Subject: No Response from Seller
Description: [Chatbot]:  Hi there, how can we help you today?   [user]:  I sent a message to the seller on 2/2 and received an auto reply to allow 2-3 days for someone to get back to me. To date, I have not heard anything from the seller.   [Chatbot]:  (No Intent Predicted)   [Chatbot]:  I understand your concern about not hearing back from the seller. If it's been more than 2 business days since you contacted them, Cratejoy can assist by reaching out on your behalf. Please contact Cratejoy Support for further help with this issue. * Shipments Lost In Transit * Getting Help With An Unshipped Order * Damaged, Duplicate or Defective Items   [Chatbot]:  Was I able to help you resolve your question?
(Yes, thank you!)
</input>

<output>
<Spam>No</Spam>
</output>
"""

    class IsTicketSpam(PromptSignature):
        ticket_summary = InputField(name="Ticket Summary", desc="A summary of the ticket", formatter=as_multiline)
        is_spam = OutputFieldBool(name="Spam", desc="Is this ticket a spam or sales ticket")

    config = {"llm_type": "anthropic"}
    prompt_runner = PromptRunner(template_class=IsTicketSpam, prompt_strategy=DefaultPromptStrategy)
    result = prompt_runner.template.parse_output_to_fields(output_data, config["llm_type"])
    
    assert result["is_spam"] == "No"

def test_output_parsing_openai_json():
    prompt_runner = PromptRunner(template_class=TestOutputParsingPromptSignature, prompt_strategy=DefaultPromptStrategy)
    
    input_data = {
        "ticket_summary": "..."
    }
    
    output_data = """
    {
        "Buyer Issues Summary": "The buyer is trying to personalize their order by selecting variants like color or size, but after making their selections and hitting \\"done\\", the changes are not being reflected. They are also asking how long delivery will take.",
        "Buyer Issue Enum": "BOX_CONTENTS_CUSTOMIZATION"
    }
    """
    
    config = {"llm_type": "openai_json"}
    result = prompt_runner.template.parse_output_to_fields(output_data, config["llm_type"])
    
    assert result["buyer_issues_summary"] == "The buyer is trying to personalize their order by selecting variants like color or size, but after making their selections and hitting \"done\", the changes are not being reflected. They are also asking how long delivery will take."
    assert result["buyer_issue_category"] == "BOX_CONTENTS_CUSTOMIZATION"