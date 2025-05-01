from typing import Any, Text, Dict, List
import psycopg2
from rasa_sdk import Action
from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet


# ✅ PostgreSQL connection config
DB_CONFIG = {
    "host": "localhost",
    "database": "rasa_chatbot",  # 🔁 Replace with your DB name
    "user": "postgres",        # 🔁 Your DB user
    "password": "satgur123",  # 🔁 Set real password
    "port": "5432"
}

# ✅ Action 1: Fetch Booking Details
class ActionFetchBookingDetails(Action):
    def name(self) -> str:
        return "action_fetch_booking_details"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        pnr = tracker.get_slot("pnr")
        customer_id = tracker.get_slot("customer_id")

        try:
            connection = psycopg2.connect(**DB_CONFIG)
            cursor = connection.cursor()
            cursor.execute(
                "SELECT miles_balance,upgrade_eligible FROM bookings WHERE pnr = %s AND customer_id = %s",
                (pnr, customer_id)
            )

            row = cursor.fetchone()
            if row:
                miles, available = row
                dispatcher.utter_message(
                    text=f"You have {miles} miles. Business class upgrade requires 30,000 miles."
                )
                return [
                    SlotSet("miles_balance", miles),
                    SlotSet("upgrade_available", available)
                ]
            else:
                dispatcher.utter_message(text="Booking not found. Please check your PNR and customer ID.")

        except Exception as e:
            dispatcher.utter_message(text=f"Database error: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()

        return []

# ✅ Action 2: Offer Cash + Miles
class ActionOfferCashMiles(Action):
    def name(self) -> str:
        return "action_offer_cash_miles"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        dispatcher.utter_message(
            text="You can upgrade using ₹8000 + 20,000 miles. Would you like to proceed?"
        )
        return []

# ✅ Action 3: Confirm and Store Upgrade
class ActionConfirmUpgrade(Action):
    def name(self) -> str:
        return "action_confirm_upgrade"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[str, Any]) -> List[Dict[str, Any]]:

        pnr = tracker.get_slot("pnr")
        customer_id = tracker.get_slot("customer_id")
        upgrade_class = tracker.get_slot("upgrade_class") or "Business"
        miles_balance = tracker.get_slot("miles_balance") or 0

        try:
            connection = psycopg2.connect(**DB_CONFIG)
            cursor = connection.cursor()
            cursor.execute("""
                INSERT INTO upgrade_requests (pnr, customer_id, upgrade_class, miles_used)
                VALUES (%s, %s, %s, %s)
            """, (pnr, customer_id, upgrade_class, miles_balance))

            connection.commit()
            dispatcher.utter_message(text="✅ Your upgrade has been confirmed and logged.")
        except Exception as e:
            dispatcher.utter_message(text=f"❌ Failed to confirm upgrade: {str(e)}")
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'connection' in locals():
                connection.close()

        return []