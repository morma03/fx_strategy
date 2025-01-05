import histdatacom
from histdatacom.options import Options
from histdatacom.fx_enums import Pairs

def import_pair_to_influx(pair, start, end, timeframes="1-minute-bar-quotes"):
    data_options = Options()

    data_options.import_to_influxdb = False  # implies validate, download, and extract
    data_options.delete_after_influx = False
    data_options.batch_size = "2000"
    data_options.cpu_utilization = "high"

    data_options.pairs = {f"{pair}"}# histdata_and_oanda_intersect_symbs
    data_options.start_yearmonth = f"{start}"
    data_options.end_yearmonth = f"{end}"
    data_options.formats = {"ascii"}  # Must be {"ascii"}
    data_options.timeframes = timeframes # can be tick-data-quotes or 1-minute-bar-quotes
    data_options.data_directory = "../../data/histdata"
    histdatacom(data_options)

# if __name__ == '__main__':
    # import_pair_to_influx("usdjpy", "202301", "202312")
