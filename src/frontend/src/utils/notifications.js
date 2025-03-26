/**
 * Utilities for handling push notifications in the PWA
 */

// Check if the browser supports notifications
export const areNotificationsSupported = () => {
  return 'Notification' in window && 'serviceWorker' in navigator && 'PushManager' in window;
};

// Request permission to show notifications
export const requestNotificationPermission = async () => {
  if (!areNotificationsSupported()) {
    console.warn('Notifications are not supported in this browser');
    return false;
  }
  
  try {
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  } catch (error) {
    console.error('Error requesting notification permission:', error);
    return false;
  }
};

// Check if permission is already granted
export const checkNotificationPermission = () => {
  if (!areNotificationsSupported()) {
    return false;
  }
  
  return Notification.permission === 'granted';
};

// Subscribe to push notifications
export const subscribeToPushNotifications = async (vapidPublicKey) => {
  if (!checkNotificationPermission()) {
    console.warn('Notification permission not granted');
    return null;
  }
  
  try {
    const registration = await navigator.serviceWorker.ready;
    
    // Get the existing subscription if there is one
    let subscription = await registration.pushManager.getSubscription();
    
    if (subscription) {
      return subscription;
    }
    
    // Otherwise, create a new subscription
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(vapidPublicKey)
    });
    
    return subscription;
  } catch (error) {
    console.error('Error subscribing to push notifications:', error);
    return null;
  }
};

// Unsubscribe from push notifications
export const unsubscribeFromPushNotifications = async () => {
  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();
    
    if (subscription) {
      await subscription.unsubscribe();
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('Error unsubscribing from push notifications:', error);
    return false;
  }
};

// Send subscription to server
export const sendSubscriptionToServer = async (subscription) => {
  if (!subscription) {
    return false;
  }
  
  try {
    const response = await fetch('/api/notifications/subscribe', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ subscription })
    });
    
    return response.ok;
  } catch (error) {
    console.error('Error sending subscription to server:', error);
    return false;
  }
};

// Helper function to convert base64 string to Uint8Array
// (needed for applicationServerKey)
function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - base64String.length % 4) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');
  
  const rawData = window.atob(base64);
  const outputArray = new Uint8Array(rawData.length);
  
  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }
  
  return outputArray;
} 