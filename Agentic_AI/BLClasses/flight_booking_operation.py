import UtilityClasses
from datetime import date

def check_if_pnr_exists(pnr_number):
    query = "select * from flight_booking_data where PNR = '{pnr_value}'".format(pnr_value=pnr_number)
    result = UtilityClasses.dbUtils.select_record(query)

    if len(result) > 0:
        return True
    else:
        return False

def get_pnr_details(pnr_number):
    query = "select * from flight_booking_data where PNR = '{pnr_value}'".format(pnr_value = pnr_number)
    print("Query to fetch PNR data - ", query)
    result = UtilityClasses.dbUtils.select_record(query)
    user_id = result[0][0]
    flight_id = result[0][1]
    travel_date = result[0][3]

    # Current system date
    today = date.today()

    query_to_fetch_user_details = "select Name from user_data where id = '{user_id}'".format(user_id = user_id)
    query_to_fetch_flight_details = "select * from flight_data where flight_Number = '{flight_number}'".format(flight_number=flight_id)

    user_data = UtilityClasses.dbUtils.select_record(query_to_fetch_user_details)
    flight_data = UtilityClasses.dbUtils.select_record(query_to_fetch_flight_details)


    if travel_date < today:
        message = "Hello {name}, you already travelled from {from_city} to {to_city} on {date}".format(
            name=user_data[0][0], from_city=flight_data[0][2],
            to_city=flight_data[0][3], date=travel_date)
    else:
        message = "Hello {name}, you are travelling from {from_city} to {to_city} on {date}".format(name = user_data[0][0], from_city = flight_data[0][2],
                                                                                      to_city = flight_data[0][3], date = travel_date.strftime('%d %B %Y'))
    return message

def cancel_ticket(pnr_number):
    query = "select * from flight_booking_data where PNR = '{pnr_value}'".format(pnr_value=pnr_number)
    delete_query = "DELETE FROM flight_booking_data WHERE PNR = '{pnr_value}'".format(pnr_value=pnr_number)
    print("Query to fetch PNR data - ", query)
    print("Query to delete PNR data - ", delete_query)
    result = UtilityClasses.dbUtils.select_record(query)
    user_id = result[0][0]
    flight_id = result[0][1]
    booking_date = result[0][2]
    travel_date = result[0][3]

    # Current system date
    today = date.today()

    if travel_date < today:
        message = "Looks like you shared expired PNR. Please check PNR again"
    else:
        delete_result = UtilityClasses.dbUtils.delete_record(delete_query)
        query_to_fetch_user_details = "select Name from user_data where id = '{user_id}'".format(user_id=user_id)
        query_to_fetch_flight_details = "select * from flight_data where flight_Number = '{flight_number}'".format(
            flight_number=flight_id)

        user_data = UtilityClasses.dbUtils.select_record(query_to_fetch_user_details)
        flight_data = UtilityClasses.dbUtils.select_record(query_to_fetch_flight_details)

        if delete_result:
            message = "Hello {name}, Ticket is successfully cancelled for booking from {from_city} to {to_city} on {date}.".format(name = user_data[0][0], from_city = flight_data[0][2],
                                                                                          to_city = flight_data[0][3], date = travel_date.strftime('%d %B %Y'))
        else:
            message = "Some technical challenge. Please try later."
    return message
