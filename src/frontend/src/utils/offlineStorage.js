/**
 * Utility functions for offline storage using IndexedDB
 */

const DB_NAME = 'SoccerPredictionDB';
const DB_VERSION = 1;
const STORES = {
  PREDICTIONS: 'predictions',
  MATCHES: 'matches',
  USER_DATA: 'userData'
};

// Initialize the database
export const initializeDB = () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onerror = (event) => {
      console.error('IndexedDB error:', event.target.error);
      reject('Error opening IndexedDB');
    };

    request.onsuccess = (event) => {
      const db = event.target.result;
      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      const db = event.target.result;

      // Create object stores with indexes
      if (!db.objectStoreNames.contains(STORES.PREDICTIONS)) {
        const predictionsStore = db.createObjectStore(STORES.PREDICTIONS, { keyPath: 'id', autoIncrement: true });
        predictionsStore.createIndex('matchId', 'matchId', { unique: false });
        predictionsStore.createIndex('timestamp', 'timestamp', { unique: false });
        predictionsStore.createIndex('synced', 'synced', { unique: false });
      }

      if (!db.objectStoreNames.contains(STORES.MATCHES)) {
        const matchesStore = db.createObjectStore(STORES.MATCHES, { keyPath: 'id' });
        matchesStore.createIndex('date', 'date', { unique: false });
        matchesStore.createIndex('homeTeam', 'homeTeam.id', { unique: false });
        matchesStore.createIndex('awayTeam', 'awayTeam.id', { unique: false });
      }

      if (!db.objectStoreNames.contains(STORES.USER_DATA)) {
        db.createObjectStore(STORES.USER_DATA, { keyPath: 'key' });
      }
    };
  });
};

// Generic function to get database connection
const getDBConnection = async () => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
};

// Generic function to add an item to a store
export const addItem = async (storeName, item) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.add(item);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error adding item to ${storeName}:`, error);
    throw error;
  }
};

// Generic function to update an item in a store
export const updateItem = async (storeName, item) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(item);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error updating item in ${storeName}:`, error);
    throw error;
  }
};

// Generic function to get an item from a store
export const getItem = async (storeName, id) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(id);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error getting item from ${storeName}:`, error);
    throw error;
  }
};

// Generic function to delete an item from a store
export const deleteItem = async (storeName, id) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(id);
      
      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error deleting item from ${storeName}:`, error);
    throw error;
  }
};

// Generic function to get all items from a store
export const getAllItems = async (storeName) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error getting all items from ${storeName}:`, error);
    throw error;
  }
};

// Function to query items using an index
export const queryByIndex = async (storeName, indexName, value) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const index = store.index(indexName);
      const request = index.getAll(value);
      
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error querying by index in ${storeName}:`, error);
    throw error;
  }
};

// Function to clear all data from a store
export const clearStore = async (storeName) => {
  try {
    const db = await getDBConnection();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.clear();
      
      request.onsuccess = () => resolve(true);
      request.onerror = () => reject(request.error);
      
      transaction.oncomplete = () => db.close();
    });
  } catch (error) {
    console.error(`Error clearing store ${storeName}:`, error);
    throw error;
  }
};

// Specialized functions for predictions
export const saveOfflinePrediction = async (prediction) => {
  // Add synced flag to mark it as not yet synced with the server
  const predictionWithMeta = {
    ...prediction,
    synced: false,
    timestamp: new Date().toISOString()
  };
  
  return addItem(STORES.PREDICTIONS, predictionWithMeta);
};

export const getUnsyncedPredictions = async () => {
  return queryByIndex(STORES.PREDICTIONS, 'synced', false);
};

export const markPredictionAsSynced = async (id) => {
  const prediction = await getItem(STORES.PREDICTIONS, id);
  if (prediction) {
    prediction.synced = true;
    return updateItem(STORES.PREDICTIONS, prediction);
  }
  return false;
};

// Specialized functions for matches
export const cacheMatches = async (matches) => {
  // First clear existing matches to avoid duplicates
  await clearStore(STORES.MATCHES);
  
  // Then add all matches
  const promises = matches.map(match => addItem(STORES.MATCHES, match));
  return Promise.all(promises);
};

export const getMatchesByDate = async (date) => {
  return queryByIndex(STORES.MATCHES, 'date', date);
};

// Function to sync data when online
export const syncOfflineData = async () => {
  if (!navigator.onLine) {
    console.log('Cannot sync while offline');
    return false;
  }
  
  try {
    // Register a background sync if supported
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
      const registration = await navigator.serviceWorker.ready;
      await registration.sync.register('sync-predictions');
      return true;
    } else {
      // Manual sync for browsers that don't support background sync
      const unsyncedPredictions = await getUnsyncedPredictions();
      
      for (const prediction of unsyncedPredictions) {
        try {
          // Send prediction to server
          const response = await fetch('/api/predictions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(prediction)
          });
          
          if (response.ok) {
            await markPredictionAsSynced(prediction.id);
          }
        } catch (error) {
          console.error('Error syncing prediction:', error);
        }
      }
      
      return true;
    }
  } catch (error) {
    console.error('Error syncing offline data:', error);
    return false;
  }
}; 