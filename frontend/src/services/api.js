/**
 * StockDNA-AI API Service
 * Centralized API client handling authentication headers, error wrapping,
 * and calls to Prediction, XAI, Data Tool, History, and Research endpoints.
 */

const API_BASE = ''; // Uses Vite proxy in development; relative in production

export const authStorage = {
  getToken: () => localStorage.getItem('stockdna_token'),
  setToken: (token) => localStorage.setItem('stockdna_token', token),
  clearToken: () => {
    localStorage.removeItem('stockdna_token');
    localStorage.removeItem('stockdna_user');
  },
  getUser: () => localStorage.getItem('stockdna_user') || 'Analyst',
  setUser: (user) => localStorage.setItem('stockdna_user', user),
  isLoggedIn: () => !!localStorage.getItem('stockdna_token'),
};

async function request(endpoint, options = {}) {
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };

  const token = authStorage.getToken();
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const config = {
    ...options,
    headers,
  };

  try {
    const response = await fetch(`${API_BASE}${endpoint}`, config);
    
    if (!response.ok) {
      let errorDetail = `HTTP ${response.status}: ${response.statusText}`;
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorDetail;
      } catch {
        // Fallback to text
      }
      throw new Error(errorDetail);
    }

    return await response.json();
  } catch (err) {
    console.error(`API Error on ${endpoint}:`, err);
    throw err;
  }
}

export const api = {
  // Authentication
  register: async (username, email, password) => {
    return await request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({ username, email, password }),
    });
  },

  login: async (email, password) => {
    const res = await request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ email, password }),
    });
    if (res.access_token) {
      authStorage.setToken(res.access_token);
      authStorage.setUser(email.split('@')[0]);
    }
    return res;
  },

  logout: () => {
    authStorage.clearToken();
  },

  // Prediction Engine & XAI
  predict: async (ticker) => {
    return await request('/predict', {
      method: 'POST',
      body: JSON.stringify({ ticker }),
    });
  },

  // Prediction History (User Scoped)
  getHistory: async () => {
    return await request('/history/');
  },

  getHistoryItem: async (predictionId) => {
    return await request(`/history/${predictionId}`);
  },

  deleteHistoryItem: async (predictionId) => {
    return await request(`/history/${predictionId}`, {
      method: 'DELETE',
    });
  },

  clearHistory: async () => {
    return await request('/history/', {
      method: 'DELETE',
    });
  },

  // Standalone Data Tool
  getUniverse: async () => {
    return await request('/datatool/universe');
  },

  classifyFactor: async (featureName) => {
    return await request('/datatool/factors/classify', {
      method: 'POST',
      body: JSON.stringify({ feature_name: featureName }),
    });
  },

  classifyFactors: async (features = []) => {
    // Call classify endpoint for list of features
    const results = await Promise.all(
      features.map(async (feat) => {
        try {
          const res = await request('/datatool/factors/classify', {
            method: 'POST',
            body: JSON.stringify({ feature_name: feat }),
          });
          return { feature: feat, group: res.category || res.group || 'EXTERNAL' };
        } catch {
          return { feature: feat, group: 'EXTERNAL' };
        }
      })
    );
    const classified = { INTERNAL: [], EXTERNAL: [] };
    results.forEach((r) => {
      if (r.group === 'INTERNAL') classified.INTERNAL.push(r.feature);
      else classified.EXTERNAL.push(r.feature);
    });
    return { classified, raw: results };
  },

  classifySentiment: async (text) => {
    return await request('/datatool/sentiment', {
      method: 'POST',
      body: JSON.stringify({ text }),
    });
  },

  processData: async (rawRecords) => {
    return await request('/datatool/process', {
      method: 'POST',
      body: JSON.stringify({ records: rawRecords }),
    });
  },

  // Research & Audited Academic Artifacts
  getResearchMetrics: async () => {
    return await request('/research/metrics');
  },

  getResearchBaselines: async () => {
    return await request('/research/baselines');
  },

  getResearchBacktest: async () => {
    return await request('/research/backtest');
  },

  getResearchRegimes: async () => {
    return await request('/research/regimes');
  },
};
