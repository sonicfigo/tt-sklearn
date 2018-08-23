# coding=utf-8
"""

"""
import json
import pprint

result_str = """
{
    "msg": "SUCCESS",
    "result":
    {
        "tripType": "OW",
        "flightSearchItemList":
        {
            "2018-10-01,XMN,PEK": [
            {
                "lowestPrice": 1240.000000,
                "dst": "PEK",
                "distance": 1774,
                "dstCN": "首都",
                "orgCity": "厦门",
                "available": true,
                "flightStopInfo": null,
                "duration": 165,
                "dstCityEn": "BEIJING",
                "operationAirlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "takeoffDate": "2018-10-01",
                "aircraftType": "788",
                "stopQuantity": null,
                "arrivalTime": "09:45",
                "codeShare": false,
                "airlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "orgTerminal": "T3",
                "dstEN": "CAPITAL INTERNATIONAL AIRPORT",
                "id": 0,
                "operatingCarrier": "MF",
                "orgEN": "GAOQI INTL AIRPORT",
                "orgCN": "高崎",
                "orgCityEn": "XIAMEN",
                "stopDuration": null,
                "org": "XMN",
                "dstTerminal": "T2",
                "soldOut": false,
                "flightNumber": "MF8117",
                "arrivalDate": "2018-10-01",
                "meal": "点心",
                "cabinCountToShow": 4,
                "carrier": "MF",
                "operatingFlightNumber": "MF8117",
                "stop": false,
                "takeoffTime": "07:00",
                "dstCity": "北京",
                "cabinInfos": [
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1240.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "Q",
                    "chdAmount": 970.0,
                    "cabin": "Q",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Q舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "Q",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(Q)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(Q)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Q)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(Q)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1240.000000,
                    "currency": "CNY",
                    "cabinNum": "2",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1930.000000,
                    "brandLevel": "7000",
                    "ei": "L变更退票收费",
                    "fbc": "Y",
                    "chdAmount": 970.0,
                    "cabin": "Y",
                    "description": "经济舱全价",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Y舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "Y",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(Y)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(Y)公布运价的10%（特殊时段为10%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Y)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(Y)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1930.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 3800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "ISWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "I",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航I舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 40,
                        "changeRateBefore": 30,
                        "cabin": "I",
                        "refundRateAfter": 80,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(I)公布运价的30%，起飞前2小时后，收取舱位(I)公布运价的40%。以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(I)公布运价的50%；起飞前2小时后收取舱位(I)公布运价的80%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 3800.000000,
                    "currency": "CNY",
                    "cabinNum": "2",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 5232,
                    "amount": 7800.000000,
                    "brandLevel": "5000",
                    "ei": "L变更退票收费",
                    "fbc": "F",
                    "chdAmount": 3900.0,
                    "cabin": "F",
                    "description": "头等舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "F",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "F",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航F舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "F",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(F)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(F)公布运价的5%（特殊时段为5%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(F)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(F)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 7800.000000,
                    "infFueTax": 0,
                    "pubPrice": 7800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1330.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "N",
                    "chdAmount": 970.0,
                    "cabin": "N",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航N舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "N",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(N)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(N)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(N)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(N)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1330.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1450.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "K",
                    "chdAmount": 970.0,
                    "cabin": "K",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航K舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "K",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(K)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(K)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(K)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(K)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1450.000000,
                    "currency": "CNY",
                    "cabinNum": "6",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1540.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "L",
                    "chdAmount": 970.0,
                    "cabin": "L",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航L舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "L",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(L)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(L)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(L)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(L)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1540.000000,
                    "currency": "CNY",
                    "cabinNum": "8",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1640.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "M",
                    "chdAmount": 970.0,
                    "cabin": "M",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航M舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "M",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(M)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(M)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。\r\n退票：航班日起飞前2小时前（含）收取舱位(M)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(M)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1640.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 2180,
                    "amount": 1830.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "B",
                    "chdAmount": 970.0,
                    "cabin": "B",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航B舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "B",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(B)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(B)公布运价的20%（特殊时段为25%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(B)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(B)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1830.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 4800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "DSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "D",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航D舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 25,
                        "changeRateBefore": 20,
                        "cabin": "D",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(D)公布运价的20%，起飞前2小时后，收取舱位(D)公布运价的25%，以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(D)公布运价的30%；起飞前2小时后收取舱位(D)公布运价的50%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 4800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 5800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "CSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "C",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航C舱退改签政策",
                        "refundRateBefore": 15,
                        "changeRateAfter": 15,
                        "changeRateBefore": 10,
                        "cabin": "C",
                        "refundRateAfter": 25,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(C)公布运价的10%，起飞前2小时后，收取舱位(C)公布运价的15%，以上改期均需补齐差价;退票：起飞前2小时前(含)收取舱位(C)公布运价的15%；起飞前2小时后收取舱位(C)公布运价的25%；来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 5800.000000,
                    "currency": "CNY",
                    "cabinNum": "6",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 6800.000000,
                    "brandLevel": "6000",
                    "ei": "L变更退票收费",
                    "fbc": "J",
                    "chdAmount": 3400.0,
                    "cabin": "J",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航J舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "J",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(J)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(J)公布运价的5%（特殊时段为5%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(J)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(J)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 6800.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                }]
            },
            {
                "lowestPrice": 1240.000000,
                "dst": "PEK",
                "distance": 1774,
                "dstCN": "首都",
                "orgCity": "厦门",
                "available": true,
                "flightStopInfo": null,
                "duration": 170,
                "dstCityEn": "BEIJING",
                "operationAirlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "takeoffDate": "2018-10-01",
                "aircraftType": "789",
                "stopQuantity": null,
                "arrivalTime": "13:50",
                "codeShare": false,
                "airlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "orgTerminal": "T3",
                "dstEN": "CAPITAL INTERNATIONAL AIRPORT",
                "id": 0,
                "operatingCarrier": "MF",
                "orgEN": "GAOQI INTL AIRPORT",
                "orgCN": "高崎",
                "orgCityEn": "XIAMEN",
                "stopDuration": null,
                "org": "XMN",
                "dstTerminal": "T2",
                "soldOut": false,
                "flightNumber": "MF8101",
                "arrivalDate": "2018-10-01",
                "meal": "正餐",
                "cabinCountToShow": 4,
                "carrier": "MF",
                "operatingFlightNumber": "MF8101",
                "stop": false,
                "takeoffTime": "11:00",
                "dstCity": "北京",
                "cabinInfos": [
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1240.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "Q",
                    "chdAmount": 970.0,
                    "cabin": "Q",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Q舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "Q",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(Q)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(Q)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Q)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(Q)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1240.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1450.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "K",
                    "chdAmount": 970.0,
                    "cabin": "K",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航K舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "K",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(K)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(K)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(K)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(K)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1450.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1930.000000,
                    "brandLevel": "7000",
                    "ei": "L变更退票收费",
                    "fbc": "Y",
                    "chdAmount": 970.0,
                    "cabin": "Y",
                    "description": "经济舱全价",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Y舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "Y",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(Y)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(Y)公布运价的10%（特殊时段为10%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Y)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(Y)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1930.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 4800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "DSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "D",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航D舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 25,
                        "changeRateBefore": 20,
                        "cabin": "D",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(D)公布运价的20%，起飞前2小时后，收取舱位(D)公布运价的25%，以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(D)公布运价的30%；起飞前2小时后收取舱位(D)公布运价的50%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 4800.000000,
                    "currency": "CNY",
                    "cabinNum": "3",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1330.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "N",
                    "chdAmount": 970.0,
                    "cabin": "N",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航N舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "N",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(N)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(N)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(N)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(N)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1330.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1540.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "L",
                    "chdAmount": 970.0,
                    "cabin": "L",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航L舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "L",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(L)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(L)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(L)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(L)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1540.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1640.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "M",
                    "chdAmount": 970.0,
                    "cabin": "M",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航M舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "M",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(M)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(M)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。\r\n退票：航班日起飞前2小时前（含）收取舱位(M)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(M)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1640.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 2180,
                    "amount": 1830.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "B",
                    "chdAmount": 970.0,
                    "cabin": "B",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航B舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "B",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(B)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(B)公布运价的20%（特殊时段为25%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(B)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(B)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1830.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1920.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "H",
                    "chdAmount": 970.0,
                    "cabin": "H",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航H舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "H",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(H)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(H)公布运价的10%（特殊时段为10%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(H)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(H)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1920.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 5800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "CSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "C",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航C舱退改签政策",
                        "refundRateBefore": 15,
                        "changeRateAfter": 15,
                        "changeRateBefore": 10,
                        "cabin": "C",
                        "refundRateAfter": 25,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(C)公布运价的10%，起飞前2小时后，收取舱位(C)公布运价的15%，以上改期均需补齐差价;退票：起飞前2小时前(含)收取舱位(C)公布运价的15%；起飞前2小时后收取舱位(C)公布运价的25%；来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 5800.000000,
                    "currency": "CNY",
                    "cabinNum": "5",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 6800.000000,
                    "brandLevel": "6000",
                    "ei": "L变更退票收费",
                    "fbc": "J",
                    "chdAmount": 3400.0,
                    "cabin": "J",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航J舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "J",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(J)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(J)公布运价的5%（特殊时段为5%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(J)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(J)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 6800.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                }]
            },
            {
                "lowestPrice": 1040.000000,
                "dst": "PEK",
                "distance": 1774,
                "dstCN": "首都",
                "orgCity": "厦门",
                "available": true,
                "flightStopInfo": null,
                "duration": 170,
                "dstCityEn": "BEIJING",
                "operationAirlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "takeoffDate": "2018-10-01",
                "aircraftType": "788",
                "stopQuantity": null,
                "arrivalTime": "17:50",
                "codeShare": false,
                "airlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "orgTerminal": "T3",
                "dstEN": "CAPITAL INTERNATIONAL AIRPORT",
                "id": 0,
                "operatingCarrier": "MF",
                "orgEN": "GAOQI INTL AIRPORT",
                "orgCN": "高崎",
                "orgCityEn": "XIAMEN",
                "stopDuration": null,
                "org": "XMN",
                "dstTerminal": "T2",
                "soldOut": false,
                "flightNumber": "MF8105",
                "arrivalDate": "2018-10-01",
                "meal": "正餐",
                "cabinCountToShow": 4,
                "carrier": "MF",
                "operatingFlightNumber": "MF8105",
                "stop": false,
                "takeoffTime": "15:00",
                "dstCity": "北京",
                "cabinInfos": [
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1040.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "V",
                    "chdAmount": 970.0,
                    "cabin": "V",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航V舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 50,
                        "changeRateBefore": 30,
                        "cabin": "V",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(V)公布运价的30%（特殊时段为35%）。起飞前2小时后收取舱位(V)公布运价的50%（特殊时段为55%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(V)公布运价的50%（特殊时段为55%），起飞前2小时后经济舱全票价(V)的100%（特殊时段为100%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1040.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1930.000000,
                    "brandLevel": "7000",
                    "ei": "L变更退票收费",
                    "fbc": "Y",
                    "chdAmount": 970.0,
                    "cabin": "Y",
                    "description": "经济舱全价",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Y舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "Y",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(Y)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(Y)公布运价的10%（特殊时段为10%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Y)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(Y)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1930.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 3800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "ISWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "I",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航I舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 40,
                        "changeRateBefore": 30,
                        "cabin": "I",
                        "refundRateAfter": 80,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(I)公布运价的30%，起飞前2小时后，收取舱位(I)公布运价的40%。以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(I)公布运价的50%；起飞前2小时后收取舱位(I)公布运价的80%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 3800.000000,
                    "currency": "CNY",
                    "cabinNum": "2",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 5232,
                    "amount": 7800.000000,
                    "brandLevel": "5000",
                    "ei": "L变更退票收费",
                    "fbc": "F",
                    "chdAmount": 3900.0,
                    "cabin": "F",
                    "description": "头等舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "F",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "F",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航F舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "F",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(F)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(F)公布运价的5%（特殊时段为5%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(F)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(F)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 7800.000000,
                    "infFueTax": 0,
                    "pubPrice": 7800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1240.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "Q",
                    "chdAmount": 970.0,
                    "cabin": "Q",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Q舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "Q",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(Q)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(Q)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Q)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(Q)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1240.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1330.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "N",
                    "chdAmount": 970.0,
                    "cabin": "N",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航N舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "N",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(N)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(N)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(N)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(N)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1330.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1450.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "K",
                    "chdAmount": 970.0,
                    "cabin": "K",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航K舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "K",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(K)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(K)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(K)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(K)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1450.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1540.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "L",
                    "chdAmount": 970.0,
                    "cabin": "L",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航L舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "L",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(L)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(L)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(L)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(L)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1540.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1640.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "M",
                    "chdAmount": 970.0,
                    "cabin": "M",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航M舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "M",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(M)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(M)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。\r\n退票：航班日起飞前2小时前（含）收取舱位(M)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(M)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1640.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 2180,
                    "amount": 1830.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "B",
                    "chdAmount": 970.0,
                    "cabin": "B",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航B舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "B",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(B)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(B)公布运价的20%（特殊时段为25%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(B)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(B)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1830.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1920.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "H",
                    "chdAmount": 970.0,
                    "cabin": "H",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航H舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "H",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(H)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(H)公布运价的10%（特殊时段为10%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(H)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(H)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1920.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 4800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "DSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "D",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航D舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 25,
                        "changeRateBefore": 20,
                        "cabin": "D",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(D)公布运价的20%，起飞前2小时后，收取舱位(D)公布运价的25%，以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(D)公布运价的30%；起飞前2小时后收取舱位(D)公布运价的50%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 4800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 5800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "CSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "C",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航C舱退改签政策",
                        "refundRateBefore": 15,
                        "changeRateAfter": 15,
                        "changeRateBefore": 10,
                        "cabin": "C",
                        "refundRateAfter": 25,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(C)公布运价的10%，起飞前2小时后，收取舱位(C)公布运价的15%，以上改期均需补齐差价;退票：起飞前2小时前(含)收取舱位(C)公布运价的15%；起飞前2小时后收取舱位(C)公布运价的25%；来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 5800.000000,
                    "currency": "CNY",
                    "cabinNum": "6",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 6800.000000,
                    "brandLevel": "6000",
                    "ei": "L变更退票收费",
                    "fbc": "J",
                    "chdAmount": 3400.0,
                    "cabin": "J",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航J舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "J",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(J)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(J)公布运价的5%（特殊时段为5%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(J)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(J)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 6800.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                }]
            },
            {
                "lowestPrice": 1040.000000,
                "dst": "PEK",
                "distance": 1774,
                "dstCN": "首都",
                "orgCity": "厦门",
                "available": true,
                "flightStopInfo": null,
                "duration": 185,
                "dstCityEn": "BEIJING",
                "operationAirlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "takeoffDate": "2018-10-01",
                "aircraftType": "757",
                "stopQuantity": null,
                "arrivalTime": "20:05",
                "codeShare": false,
                "airlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "orgTerminal": "T3",
                "dstEN": "CAPITAL INTERNATIONAL AIRPORT",
                "id": 0,
                "operatingCarrier": "MF",
                "orgEN": "GAOQI INTL AIRPORT",
                "orgCN": "高崎",
                "orgCityEn": "XIAMEN",
                "stopDuration": null,
                "org": "XMN",
                "dstTerminal": "T2",
                "soldOut": false,
                "flightNumber": "MF8169",
                "arrivalDate": "2018-10-01",
                "meal": "点心",
                "cabinCountToShow": 4,
                "carrier": "MF",
                "operatingFlightNumber": "MF8169",
                "stop": false,
                "takeoffTime": "17:00",
                "dstCity": "北京",
                "cabinInfos": [
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1040.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "V",
                    "chdAmount": 970.0,
                    "cabin": "V",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航V舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 50,
                        "changeRateBefore": 30,
                        "cabin": "V",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(V)公布运价的30%（特殊时段为35%）。起飞前2小时后收取舱位(V)公布运价的50%（特殊时段为55%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(V)公布运价的50%（特殊时段为55%），起飞前2小时后经济舱全票价(V)的100%（特殊时段为100%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1040.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1930.000000,
                    "brandLevel": "7000",
                    "ei": "L变更退票收费",
                    "fbc": "Y",
                    "chdAmount": 970.0,
                    "cabin": "Y",
                    "description": "经济舱全价",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Y舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "Y",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(Y)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(Y)公布运价的10%（特殊时段为10%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Y)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(Y)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1930.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 3800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "ISWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "I",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航I舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 40,
                        "changeRateBefore": 30,
                        "cabin": "I",
                        "refundRateAfter": 80,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(I)公布运价的30%，起飞前2小时后，收取舱位(I)公布运价的40%。以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(I)公布运价的50%；起飞前2小时后收取舱位(I)公布运价的80%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 3800.000000,
                    "currency": "CNY",
                    "cabinNum": "2",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 5232,
                    "amount": 7800.000000,
                    "brandLevel": "5000",
                    "ei": "L变更退票收费",
                    "fbc": "F",
                    "chdAmount": 3900.0,
                    "cabin": "F",
                    "description": "头等舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "F",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "F",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航F舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "F",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(F)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(F)公布运价的5%（特殊时段为5%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(F)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(F)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 7800.000000,
                    "infFueTax": 0,
                    "pubPrice": 7800.000000,
                    "currency": "CNY",
                    "cabinNum": "8",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1240.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "Q",
                    "chdAmount": 970.0,
                    "cabin": "Q",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Q舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "Q",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(Q)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(Q)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Q)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(Q)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1240.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1330.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "N",
                    "chdAmount": 970.0,
                    "cabin": "N",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航N舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "N",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(N)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(N)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(N)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(N)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1330.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1450.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "K",
                    "chdAmount": 970.0,
                    "cabin": "K",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航K舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "K",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(K)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(K)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(K)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(K)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1450.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1540.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "L",
                    "chdAmount": 970.0,
                    "cabin": "L",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航L舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "L",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(L)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(L)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(L)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(L)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1540.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1640.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "M",
                    "chdAmount": 970.0,
                    "cabin": "M",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航M舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "M",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(M)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(M)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。\r\n退票：航班日起飞前2小时前（含）收取舱位(M)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(M)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1640.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 2180,
                    "amount": 1830.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "B",
                    "chdAmount": 970.0,
                    "cabin": "B",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航B舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "B",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(B)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(B)公布运价的20%（特殊时段为25%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(B)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(B)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1830.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1920.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "H",
                    "chdAmount": 970.0,
                    "cabin": "H",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航H舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "H",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(H)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(H)公布运价的10%（特殊时段为10%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(H)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(H)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1920.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 4800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "DSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "D",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航D舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 25,
                        "changeRateBefore": 20,
                        "cabin": "D",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(D)公布运价的20%，起飞前2小时后，收取舱位(D)公布运价的25%，以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(D)公布运价的30%；起飞前2小时后收取舱位(D)公布运价的50%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 4800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 5800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "CSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "C",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航C舱退改签政策",
                        "refundRateBefore": 15,
                        "changeRateAfter": 15,
                        "changeRateBefore": 10,
                        "cabin": "C",
                        "refundRateAfter": 25,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(C)公布运价的10%，起飞前2小时后，收取舱位(C)公布运价的15%，以上改期均需补齐差价;退票：起飞前2小时前(含)收取舱位(C)公布运价的15%；起飞前2小时后收取舱位(C)公布运价的25%；来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 5800.000000,
                    "currency": "CNY",
                    "cabinNum": "6",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 6800.000000,
                    "brandLevel": "6000",
                    "ei": "L变更退票收费",
                    "fbc": "J",
                    "chdAmount": 3400.0,
                    "cabin": "J",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航J舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "J",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(J)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(J)公布运价的5%（特殊时段为5%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(J)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(J)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 6800.000000,
                    "currency": "CNY",
                    "cabinNum": "8",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                }]
            },
            {
                "lowestPrice": 850.000000,
                "dst": "PEK",
                "distance": 1774,
                "dstCN": "首都",
                "orgCity": "厦门",
                "available": true,
                "flightStopInfo": null,
                "duration": 170,
                "dstCityEn": "BEIJING",
                "operationAirlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "takeoffDate": "2018-10-01",
                "aircraftType": "738",
                "stopQuantity": null,
                "arrivalTime": "00:20",
                "codeShare": false,
                "airlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "orgTerminal": "T3",
                "dstEN": "CAPITAL INTERNATIONAL AIRPORT",
                "id": 0,
                "operatingCarrier": "MF",
                "orgEN": "GAOQI INTL AIRPORT",
                "orgCN": "高崎",
                "orgCityEn": "XIAMEN",
                "stopDuration": null,
                "org": "XMN",
                "dstTerminal": "T2",
                "soldOut": false,
                "flightNumber": "MF8103",
                "arrivalDate": "2018-10-02",
                "meal": "点心",
                "cabinCountToShow": 4,
                "carrier": "MF",
                "operatingFlightNumber": "MF8103",
                "stop": false,
                "takeoffTime": "21:30",
                "dstCity": "北京",
                "cabinInfos": [
                {
                    "reduce": false,
                    "cabinPoint": null,
                    "amount": 850.000000,
                    "brandLevel": "4000",
                    "ei": "特价/不得退改签升舱",
                    "fbc": "RPR084",
                    "chdAmount": 850.000000,
                    "cabin": "R",
                    "description": "特价专享",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": null,
                    "refundRules":
                    {
                        "cabinStandard": "R",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航R舱退改签政策",
                        "refundRateBefore": 100,
                        "changeRateAfter": 100,
                        "changeRateBefore": 100,
                        "cabin": "R",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：不允许；退票：仅退还未使用航段相关税费；"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 850.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "RUZ",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": null,
                    "amount": 950.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "T",
                    "chdAmount": 950.000000,
                    "cabin": "T",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": null,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航T舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 50,
                        "changeRateBefore": 30,
                        "cabin": "T",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(T)公布运价的30%（特殊时段为35%）。起飞前2小时后收取舱位(T)公布运价的50%（特殊时段为55%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(T)公布运价的50%（特殊时段为55%），起飞前2小时后收取舱位(T)公布运价的100%（特殊时段为100%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 950.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1930.000000,
                    "brandLevel": "7000",
                    "ei": "L变更退票收费",
                    "fbc": "Y",
                    "chdAmount": 970.0,
                    "cabin": "Y",
                    "description": "经济舱全价",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Y舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "Y",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(Y)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(Y)公布运价的10%（特殊时段为10%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Y)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(Y)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1930.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 4800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "DSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "D",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航D舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 25,
                        "changeRateBefore": 20,
                        "cabin": "D",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(D)公布运价的20%，起飞前2小时后，收取舱位(D)公布运价的25%，以上改期均需补齐差价；退票：起飞前2小时前(含)收取舱位(D)公布运价的30%；起飞前2小时后收取舱位(D)公布运价的50%。来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 4800.000000,
                    "currency": "CNY",
                    "cabinNum": "2",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1040.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "V",
                    "chdAmount": 970.0,
                    "cabin": "V",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航V舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 50,
                        "changeRateBefore": 30,
                        "cabin": "V",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(V)公布运价的30%（特殊时段为35%）。起飞前2小时后收取舱位(V)公布运价的50%（特殊时段为55%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(V)公布运价的50%（特殊时段为55%），起飞前2小时后经济舱全票价(V)的100%（特殊时段为100%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1040.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1240.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "Q",
                    "chdAmount": 970.0,
                    "cabin": "Q",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Q舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "Q",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(Q)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(Q)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Q)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(Q)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1240.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1330.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "N",
                    "chdAmount": 970.0,
                    "cabin": "N",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航N舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "N",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(N)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(N)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(N)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(N)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1330.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1450.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "K",
                    "chdAmount": 970.0,
                    "cabin": "K",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航K舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "K",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(K)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(K)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(K)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(K)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1450.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1540.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "L",
                    "chdAmount": 970.0,
                    "cabin": "L",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航L舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "L",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(L)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(L)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(L)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(L)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1540.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1640.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "M",
                    "chdAmount": 970.0,
                    "cabin": "M",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航M舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "M",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(M)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(M)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。\r\n退票：航班日起飞前2小时前（含）收取舱位(M)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(M)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1640.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 2180,
                    "amount": 1830.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "B",
                    "chdAmount": 970.0,
                    "cabin": "B",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航B舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "B",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(B)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(B)公布运价的20%（特殊时段为25%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(B)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(B)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1830.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 5800.000000,
                    "brandLevel": "6000",
                    "ei": "变更退票收费不得签转",
                    "fbc": "CSWC102A",
                    "chdAmount": 3400.0,
                    "cabin": "C",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航C舱退改签政策",
                        "refundRateBefore": 15,
                        "changeRateAfter": 15,
                        "changeRateBefore": 10,
                        "cabin": "C",
                        "refundRateAfter": 25,
                        "ruleDescription": "改期：起飞前2小时前(含)收取舱位(C)公布运价的10%，起飞前2小时后，收取舱位(C)公布运价的15%，以上改期均需补齐差价;退票：起飞前2小时前(含)收取舱位(C)公布运价的15%；起飞前2小时后收取舱位(C)公布运价的25%；来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 5800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 4360,
                    "amount": 6800.000000,
                    "brandLevel": "6000",
                    "ei": "L变更退票收费",
                    "fbc": "J",
                    "chdAmount": 3400.0,
                    "cabin": "J",
                    "description": "商务舱",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "J",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "J",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航J舱退改签政策",
                        "refundRateBefore": 5,
                        "changeRateAfter": 5,
                        "changeRateBefore": 5,
                        "cabin": "J",
                        "refundRateAfter": 10,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(J)公布运价的5%（特殊时段免费改期3次，超过3次收取5%）。起飞前2小时后收取舱位(J)公布运价的5%（特殊时段为5%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(J)公布运价的5%（特殊时段为5%），起飞前2小时后收取舱位(J)公布运价的10%（特殊时段为10%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 6800.000000,
                    "infFueTax": 0,
                    "pubPrice": 6800.000000,
                    "currency": "CNY",
                    "cabinNum": "4",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                }]
            },
            {
                "lowestPrice": 850.000000,
                "dst": "PEK",
                "distance": 1774,
                "dstCN": "首都",
                "orgCity": "厦门",
                "available": true,
                "flightStopInfo": [
                {
                    "takeOffTime": "2018-10-01 17:35",
                    "dstCityCN": "舟山",
                    "arrivalTime": "2018-10-01 19:10",
                    "planeModel": "738",
                    "dstCountry": null,
                    "orgCity": "XMN",
                    "orgCountry": null,
                    "orgCityCN": "厦门",
                    "dstCity": "HSN"
                },
                {
                    "takeOffTime": "2018-10-01 19:55",
                    "dstCityCN": "北京",
                    "arrivalTime": "2018-10-01 22:25",
                    "planeModel": "738",
                    "dstCountry": null,
                    "orgCity": "HSN",
                    "orgCountry": null,
                    "orgCityCN": "舟山",
                    "dstCity": "PEK"
                }],
                "duration": 290,
                "dstCityEn": "BEIJING",
                "operationAirlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "takeoffDate": "2018-10-01",
                "aircraftType": "738",
                "stopQuantity": null,
                "arrivalTime": "22:25",
                "codeShare": false,
                "airlineInfo":
                {
                    "code": "MF",
                    "fullName": "厦门航空",
                    "shortName": "厦航"
                },
                "orgTerminal": "T3",
                "dstEN": "CAPITAL INTERNATIONAL AIRPORT",
                "id": 0,
                "operatingCarrier": "MF",
                "orgEN": "GAOQI INTL AIRPORT",
                "orgCN": "高崎",
                "orgCityEn": "XIAMEN",
                "stopDuration": [
                    45
                ],
                "org": "XMN",
                "dstTerminal": "T2",
                "soldOut": false,
                "flightNumber": "MF8159",
                "arrivalDate": "2018-10-01",
                "meal": "正餐",
                "cabinCountToShow": 4,
                "carrier": "MF",
                "operatingFlightNumber": "MF8159",
                "stop": true,
                "takeoffTime": "17:35",
                "dstCity": "北京",
                "cabinInfos": [
                {
                    "reduce": false,
                    "cabinPoint": null,
                    "amount": 850.000000,
                    "brandLevel": "4000",
                    "ei": "特价/不得退改签升舱",
                    "fbc": "RPR084",
                    "chdAmount": 850.000000,
                    "cabin": "R",
                    "description": "特价专享",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": null,
                    "refundRules":
                    {
                        "cabinStandard": "R",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航R舱退改签政策",
                        "refundRateBefore": 100,
                        "changeRateAfter": 100,
                        "changeRateBefore": 100,
                        "cabin": "R",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：不允许；退票：仅退还未使用航段相关税费；"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 850.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "RUZ",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": null,
                    "amount": 950.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "T",
                    "chdAmount": 950.000000,
                    "cabin": "T",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": null,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航T舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 50,
                        "changeRateBefore": 30,
                        "cabin": "T",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(T)公布运价的30%（特殊时段为35%）。起飞前2小时后收取舱位(T)公布运价的50%（特殊时段为55%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(T)公布运价的50%（特殊时段为55%），起飞前2小时后收取舱位(T)公布运价的100%（特殊时段为100%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 950.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1450.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "K",
                    "chdAmount": 970.0,
                    "cabin": "K",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航K舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "K",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(K)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(K)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(K)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(K)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1450.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 3488,
                    "amount": 1930.000000,
                    "brandLevel": "7000",
                    "ei": "L变更退票收费",
                    "fbc": "Y",
                    "chdAmount": 970.0,
                    "cabin": "Y",
                    "description": "经济舱全价",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Y舱退改签政策",
                        "refundRateBefore": 10,
                        "changeRateAfter": 10,
                        "changeRateBefore": 10,
                        "cabin": "Y",
                        "refundRateAfter": 15,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)免费改期3次，超过3次收取舱位(Y)公布运价的10%（特殊时段免费改期3次，超过3次收取10%）。起飞前2小时后收取舱位(Y)公布运价的10%（特殊时段为10%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Y)公布运价的10%（特殊时段为10%），起飞前2小时后收取舱位(Y)公布运价的15%（特殊时段为15%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1930.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1040.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "V",
                    "chdAmount": 970.0,
                    "cabin": "V",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航V舱退改签政策",
                        "refundRateBefore": 50,
                        "changeRateAfter": 50,
                        "changeRateBefore": 30,
                        "cabin": "V",
                        "refundRateAfter": 100,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(V)公布运价的30%（特殊时段为35%）。起飞前2小时后收取舱位(V)公布运价的50%（特殊时段为55%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(V)公布运价的50%（特殊时段为55%），起飞前2小时后经济舱全票价(V)的100%（特殊时段为100%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1040.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1240.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "Q",
                    "chdAmount": 970.0,
                    "cabin": "Q",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航Q舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "Q",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(Q)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(Q)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(Q)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(Q)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1240.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 872,
                    "amount": 1330.000000,
                    "brandLevel": "3000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "N",
                    "chdAmount": 970.0,
                    "cabin": "N",
                    "description": "超值优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航N舱退改签政策",
                        "refundRateBefore": 30,
                        "changeRateAfter": 30,
                        "changeRateBefore": 20,
                        "cabin": "N",
                        "refundRateAfter": 50,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(N)公布运价的20%（特殊时段为25%）。起飞前2小时后收取舱位(N)公布运价的30%（特殊时段为35%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(N)公布运价的30%（特殊时段为35%），起飞前2小时后收取舱位(N)公布运价的50%（特殊时段为55%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1330.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1540.000000,
                    "brandLevel": "2000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "L",
                    "chdAmount": 970.0,
                    "cabin": "L",
                    "description": "快乐常飞",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航L舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "L",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(L)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(L)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(L)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(L)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1540.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 1744,
                    "amount": 1640.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "M",
                    "chdAmount": 970.0,
                    "cabin": "M",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航M舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "M",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(M)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(M)公布运价的20%（特殊时段为25%）。以上改期均需补齐差价。\r\n退票：航班日起飞前2小时前（含）收取舱位(M)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(M)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1640.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                },
                {
                    "reduce": false,
                    "cabinPoint": 2180,
                    "amount": 1830.000000,
                    "brandLevel": "1000",
                    "ei": "L变更退票收费不得签转",
                    "fbc": "B",
                    "chdAmount": 970.0,
                    "cabin": "B",
                    "description": "经济舱优惠",
                    "airportTax": 50.000000,
                    "infAirportTax": 0,
                    "chdFuelTax": 0,
                    "cabinLevel": "Y",
                    "fuelTax": 10.000000,
                    "cabinExtraPoint": 0,
                    "refundRules":
                    {
                        "cabinStandard": "Y",
                        "freeTimes": 0,
                        "rateType": 2,
                        "timeLength": 2,
                        "ei": "厦航B舱退改签政策",
                        "refundRateBefore": 20,
                        "changeRateAfter": 20,
                        "changeRateBefore": 10,
                        "cabin": "B",
                        "refundRateAfter": 30,
                        "ruleDescription": "改期：航班日起飞前2小时前(含)收取舱位(B)公布运价的10%（特殊时段为15%）。起飞前2小时后收取舱位(B)公布运价的20%（特殊时段为25%），以上改期均需补齐差价。退票：航班日起飞前2小时前（含）收取舱位(B)公布运价的20%（特殊时段为25%），起飞前2小时后收取舱位(B)公布运价的30%（特殊时段为35%）；此外，来回程客票，若部分使用，应先返还已享受的优惠，同时收取舱位对应的退票费。\r\n（注：特殊时段是指航班日为每年春运、7月1日-8月31日期间）"
                    },
                    "chdAirportTax": 0,
                    "fbcOverride": 1930.000000,
                    "infFueTax": 0,
                    "pubPrice": 1830.000000,
                    "currency": "CNY",
                    "cabinNum": "A",
                    "rtCabinRule": "FJCDIYBMLKNQVTH",
                    "productList": [
                    {
                        "productID": "9fe469eb114d4cfcbe35790493c9620b",
                        "productName": "航空延误险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115003",
                        "createTime": 1515980173000,
                        "productTypeKind": "030101",
                        "price": 20,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "1"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "delay_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927001",
                            "price": 20,
                            "propertyList": null,
                            "resourceName": "航空延误险",
                            "resourceType": "insurance"
                        }]
                    },
                    {
                        "productID": "77b0646e7f9745f1be1c0e9f891f6e51",
                        "productName": "航空意外险",
                        "productDesc": null,
                        "refundDesc": null,
                        "productCode": "P20180115009",
                        "createTime": 1515996286000,
                        "productTypeKind": null,
                        "price": 30,
                        "restNum": null,
                        "refundable": "1",
                        "properties": [
                        {
                            "propertyCode": "soldNum",
                            "propertyValue": "2"
                        }],
                        "productType": "0301",
                        "resourceList": [
                        {
                            "resourceTypeKind": "accident_ins",
                            "resourceDesc": null,
                            "resourceCode": "R20170927002",
                            "price": 30,
                            "propertyList": null,
                            "resourceName": "航空意外险",
                            "resourceType": "insurance"
                        }]
                    }]
                }]
            }]
        }
    },
    "code": 0
}
"""

rp_content = json.loads(result_str.replace('\r\n', ''))

flights = rp_content['result']['flightSearchItemList']['2018-10-01,XMN,PEK']


def extract_flights():
    simple_flights = {}
    for flight in flights:
        flight_no = flight['operatingFlightNumber']
        simple_flights[flight_no] = {}  # {'MF8117':{}, 'MF8101':{}}

        cabin_infos = flight['cabinInfos']
        simple_cabin_infos = _extract_simple_cabin_infos(cabin_infos)
        simple_flights[flight_no]['simple_cabins'] = simple_cabin_infos
    return simple_flights


def _extract_simple_cabin_infos(cabin_infos):
    simple_cabin_infos = []
    for cabin_info in cabin_infos:
        simple_cabin_infos.append({
            'cabin_text': cabin_info['cabin'],
            'pub_price': cabin_info['pubPrice']
        })
    return simple_cabin_infos


flights = extract_flights()
pprint.pprint(flights)
