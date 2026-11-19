from agents.geological_agent import GeologicalAgent

test_well = {
    "well_id": "TEST-001",
    "basin": "Permian", "formation": "Wolfcamp",
    "depth_m": 2800, "net_pay_m": 25, "porosity_pct": 8.5,
    "permeability_md": 0.4, "water_saturation_pct": 30,
    "skin_factor": 2.0, "stimulation_type": "acidizing",
    "proppant_tonnes": 0, "acid_volume_m3": 20, "wti_price_usd": 65.0
}

result = GeologicalAgent().run(test_well)
print(f"Reservoir Score: {result['reservoir_score']:.1f}/100")
print(f"Risk Level: {result['risk_level']}")
print(f"Analog EUR range: P10={result['analog_eur_stats']['p10']:.3f} | P50={result['analog_eur_stats']['p50']:.3f} | P90={result['analog_eur_stats']['p90']:.3f} MMBOE")
for a in result['analog_wells']:
    print(f"  - {a['well_name']} | EUR: {a['eur_mmboe']:.3f} MMBOE")
