/**
 * Authoritative 22-Equity Universe for StockDNA-AI
 * Sourced from data/reference/companies.csv across 11 key market sectors.
 */

export const DEFAULT_UNIVERSE = [
  { ticker: 'TCS', name: 'Tata Consultancy Services', sector: 'IT' },
  { ticker: 'INFY', name: 'Infosys', sector: 'IT' },
  { ticker: 'RELIANCE', name: 'Reliance Industries', sector: 'Energy' },
  { ticker: 'WIPRO', name: 'Wipro', sector: 'IT' },
  { ticker: 'HCLTECH', name: 'HCL Technologies', sector: 'IT' },
  { ticker: 'HDFCBANK', name: 'HDFC Bank', sector: 'Banking' },
  { ticker: 'ICICIBANK', name: 'ICICI Bank', sector: 'Banking' },
  { ticker: 'HINDUNILVR', name: 'Hindustan Unilever', sector: 'FMCG' },
  { ticker: 'ITC', name: 'ITC', sector: 'FMCG' },
  { ticker: 'NESTLEIND', name: 'Nestle India', sector: 'FMCG' },
  { ticker: 'MARUTI', name: 'Maruti Suzuki', sector: 'Auto' },
  { ticker: 'M&M', name: 'Mahindra & Mahindra', sector: 'Auto' },
  { ticker: 'SUNPHARMA', name: 'Sun Pharma', sector: 'Pharma' },
  { ticker: 'DRREDDY', name: "Dr. Reddy's Laboratories", sector: 'Pharma' },
  { ticker: 'TATASTEEL', name: 'Tata Steel', sector: 'Metals' },
  { ticker: 'ONGC', name: 'ONGC', sector: 'Energy' },
  { ticker: 'NTPC', name: 'NTPC', sector: 'Power' },
  { ticker: 'POWERGRID', name: 'Power Grid Corporation', sector: 'Power' },
  { ticker: 'LT', name: 'Larsen & Toubro', sector: 'Infrastructure' },
  { ticker: 'ULTRACEMCO', name: 'UltraTech Cement', sector: 'Cement' },
  { ticker: 'BHARTIARTL', name: 'Bharti Airtel', sector: 'Telecom' },
  { ticker: 'ASIANPAINT', name: 'Asian Paints', sector: 'Consumer' },
];

export const SECTORS = [
  'All',
  'IT',
  'Banking',
  'FMCG',
  'Auto',
  'Pharma',
  'Energy',
  'Power',
  'Metals',
  'Infrastructure',
  'Cement',
  'Telecom',
  'Consumer',
];

/**
 * Normalizes backend universe records to standard format
 */
export function normalizeUniverse(rawRecords) {
  if (!Array.isArray(rawRecords) || rawRecords.length === 0) {
    return DEFAULT_UNIVERSE;
  }
  return rawRecords.map((r) => {
    const rawTicker = r.ticker || r.clean_ticker || r.Ticker || '';
    const cleanTicker = rawTicker.replace('.NS', '').trim().toUpperCase();
    return {
      ticker: cleanTicker,
      name: r.company || r.Company || cleanTicker,
      sector: r.sector || r.Sector || 'General',
    };
  });
}

