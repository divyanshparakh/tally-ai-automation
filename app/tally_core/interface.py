import logging
import re
import xml.etree.ElementTree as ET
from typing import List, Optional
from tally_integration import TallyClient, TallyConnectionError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

class TallyInterface:
    """
    Core Tally ERP service wrapper.
    Handles connection, XML execution, and basic data retrieval.
    """

    def __init__(self, host: str = "http://localhost", port: int = 9000, timeout: int = 90):
        """Initialize Tally service."""
        
        try:
            self.client = TallyClient(tally_url=host, tally_port=port, timeout=timeout)
            logger.info(f"Connected to Tally at {host}:{port}")
        except TallyConnectionError as e:
            logger.error("Failed to connect to Tally: %s", e)
            self.client = None

    def check_connection(self) -> bool:
        """Checks if Tally is accessible."""
        if not self.client:
            return False
        try:
            return self.client.test_connection()
        except Exception:
            return False

    def clean_xml_response(self, xml_string: str) -> str:
        """
        Removes illegal XML numeric entities that cause parsing errors.
        """
        def replace_entity(match):
            entity = match.group(0)
            code_str = match.group(1)
            try:
                if code_str.lower().startswith('x'):
                    code = int(code_str[1:], 16)
                else:
                    code = int(code_str)
                
                # XML 1.0 Illegal characters range
                if (0 <= code <= 8) or (code in [11, 12]) or (14 <= code <= 31):
                    return ""
                return entity
            except ValueError:
                return entity

        pattern = re.compile(r'&#(x?[0-9a-fA-F]+);')
        return pattern.sub(replace_entity, xml_string)

    def execute_xml_query(self, xml_payload: str) -> str:
        """Sends arbitrary Tally XML payload and returns the response."""
        if not self.client:
            return ""
        
        try:
            response = self.client._send_request(xml_request=xml_payload)
            cleaned_xml = self.clean_xml_response(response.strip())
            return cleaned_xml
        except Exception as e:
            logger.error("Error executing Tally XML: %s", e)
            return ""

    def get_current_company_detail(self) -> Optional[str]:
        """Fetches the active company name from Tally."""
        if not self.client:
            return None
        
        try:
            raw_xml = self.client.get_current_company()
            cleaned_xml = self.clean_xml_response(raw_xml.strip())
            root = ET.fromstring(cleaned_xml)

            for node in root.findall('.//CURRENTCOMPANY'):
                if node.text and node.text.strip():
                    return node.text.strip()
            return None
        except Exception as e:
            logger.error("Error fetching company: %s", e)
            return None

    def get_all_stock_items(self) -> List[str]:
        """Retrieves a list of all stock item names."""
        if not self.client:
            return []
        
        xml_payload = """
        <ENVELOPE>
            <HEADER>
                <VERSION>1</VERSION>
                <TALLYREQUEST>Export</TALLYREQUEST>
                <TYPE>Collection</TYPE>
                <ID>maincol</ID>
            </HEADER>
            <BODY>
                <DESC>
                    <STATICVARIABLES>
                        <SVEXPORTFORMAT>$$SysName:XML</SVEXPORTFORMAT>
                    </STATICVARIABLES>
                    <TDL>
                        <TDLMESSAGE>
                            <COLLECTION NAME="maincol">
                                <Type>StocKItem</Type>
                                <FETCH>Name</FETCH>
                            </COLLECTION>
                        </TDLMESSAGE>
                    </TDL>
                </DESC>
            </BODY>
        </ENVELOPE>"""

        try:
            response = self.execute_xml_query(xml_payload)
            root = ET.fromstring(response)
            stock_items = [item.get('NAME') for item in root.iter('STOCKITEM') if item.get('NAME')]
            return stock_items
        except Exception as e:
            logger.error("Error fetching stock items: %s", e)
            return []

    def get_all_ledgers_list(self) -> List[str]:
        """Retrieves a list of all party ledger names."""
        if not self.client:
            return []
        
        try:
            response = self.client.get_ledgers_list()
            cleaned_xml = self.clean_xml_response(response.strip())
            root = ET.fromstring(cleaned_xml)
            return [item.get('NAME') for item in root.iter('LEDGER') if item.get('NAME')]
        except Exception as e:
            logger.error("Error fetching party ledgers: %s", e)
            return []
