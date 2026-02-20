from tally_core.interface import TallyInterface
import os

tallyInterface = TallyInterface(os.getenv("TALLY_HOST", "http://localhost"), int(os.getenv("TALLY_PORT", "9000")))
print(tallyInterface.get_all_ledgers_list())
print(tallyInterface.get_all_stock_items())
print(tallyInterface.get_current_company_detail())
